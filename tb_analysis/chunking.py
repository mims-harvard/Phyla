from phyla import phyla
import pandas as pd
import torch
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import yaml
import glob
from skbio import DistanceMatrix
from skbio.tree import nj
import colorsys
import pickle

with open("tb_analysis/config.yaml", "r") as file:
    config = yaml.safe_load(file)

fasta_file_path = config.get("fasta_file_path", "")
df = pd.read_csv(config.get("metadata_file_path", ""), sep="\t")
chunk_size = config.get("chunk_size", 500)
fasta_files = glob.glob(os.path.join(fasta_file_path, "**", "*.fasta"), recursive=True)

id_to_sequence = {}
sequence_length = 0
for fasta_file in tqdm(fasta_files, total=len(fasta_files), desc="Reading FASTA files"):
    with open(fasta_file, "r") as file:
        lines = file.readlines()
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
        id_to_sequence[fasta_file.split('/')[-1].split('.')[0]] = sequence
        sequence_length = max(sequence_length, len(sequence))

id_sequence_chunk = {}
for id in tqdm(id_to_sequence, total=len(id_to_sequence), desc="Chunking sequences"):
    seq_len = len(id_to_sequence[id])
    for chunk_id in range(0, (seq_len + chunk_size - 1) // chunk_size):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, seq_len)
        chunk_seq = id_to_sequence[id][start:end]
        id_sequence_chunk[(id, chunk_id)] = chunk_seq.ljust(chunk_size, 'N')

id_to_lineage = {}
for id in id_to_sequence:
    if id in df['SampleID'].values:
        lineage_value = df.loc[df['SampleID'] == id, 'PrimaryLineage'].values[0]
        id_to_lineage[id] = lineage_value
    else:
        print(f"Warning: {id} not found in metadata.")
        import pdb; pdb.set_trace()

num_files = len(fasta_files)
sequences = []

# Establish a stable ID order across chunks
ids_list = list(id_to_sequence.keys())
num_ids = len(ids_list)
max_num_chunks = (sequence_length + chunk_size - 1) // chunk_size
if 'averaged_embeddings.pt' not in os.listdir():
    output_sum = None
    num_outputs = 0
    model = phyla(name='phyla-beta').load().cuda()
    model.eval()

    print("Running inference on each concatenation of chunks...")



    for i in tqdm(range(max_num_chunks)):
        with torch.no_grad():
            concats = []
            present_idx = []
            for j, id in enumerate(ids_list):
                key = (id, i)
                if key in id_sequence_chunk:
                    concats.append(id_sequence_chunk[key])
                    present_idx.append(j)

            # If no sequences have this chunk, skip this iteration
            if len(concats) == 0:
                continue

            encoded_aa, cls_token_mask, sequence_mask, sequence_names = model.encode(concats, None)
            preds = model(encoded_aa.cuda(), sequence_mask.cuda(), cls_token_mask.cuda()).squeeze(0)

            # Build a full-size prediction tensor with zeros for missing IDs
            full_shape = (num_ids,) + tuple(preds.shape[1:])
            preds_full = torch.zeros(full_shape, device=preds.device, dtype=preds.dtype)
            idx_tensor = torch.as_tensor(present_idx, device=preds.device, dtype=torch.long)
            preds_full.index_copy_(0, idx_tensor, preds)

        if output_sum is None:
            output_sum = preds_full
        else:
            output_sum += preds_full
        num_outputs += 1

        del encoded_aa, cls_token_mask, sequence_mask, sequence_names, preds, preds_full
        gc.collect()
        torch.cuda.empty_cache()

    final_output = output_sum / num_outputs
    final_output = final_output.cpu()
    torch.save(final_output, "averaged_embeddings.pt")
    print(f"Saved averaged embeddings with shape: {final_output.shape}")
else:
    final_output = torch.load("averaged_embeddings.pt")
    print(f"Loaded averaged embeddings with shape: {final_output.shape}")

sequence_embeddings = final_output
distance_matrix = torch.cdist(sequence_embeddings, sequence_embeddings, compute_mode='donot_use_mm_for_euclid_dist').cpu().detach().numpy()
if distance_matrix.dtype != float:
    distance_matrix = distance_matrix.astype(float)
# Reconstruct tree using scikit bio

dm = DistanceMatrix(distance_matrix, ids_list)
tree = nj(dm)
with open("tb_analysis/neighbor_joining_tree.nwk", "w") as file:
    tree.write(file, format='newick')

# Build iTOL TREE_COLORS dataset to color leaf labels by lineage
lineages = sorted(set(id_to_lineage.get(i, 'UNKNOWN') for i in ids_list))
n = max(1, len(lineages))

def hsv_hex(h, s=0.65, v=0.95):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

# Evenly spaced hues for visually distinct colors
lineage_to_hex = {lin: hsv_hex(idx / n) for idx, lin in enumerate(lineages)}
lineage_to_hex['UNKNOWN'] = '#999999'

itol_path = 'tb_analysis/lineage_colors.txt'
with open(itol_path, 'w') as fh:
    fh.write('TREE_COLORS\n')
    fh.write('SEPARATOR SPACE\n')
    fh.write('DATA\n')

    # One line per leaf: color its label by lineage
    for sid in ids_list:
        lin = id_to_lineage.get(sid, 'UNKNOWN')
        color = lineage_to_hex.get(lin, lineage_to_hex['UNKNOWN'])
        fh.write(f"{sid} label {color}\n")

# Save lineage->color mapping for reuse
with open('tb_analysis/lineage_color_mapping.pkl', 'wb') as f:
    pickle.dump(lineage_to_hex, f)

print(f"Wrote iTOL TREE_COLORS file to {itol_path} with {len(ids_list)} leaf entries and {len(lineages)} lineages.")