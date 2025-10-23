from phyla import phyla
import pandas as pd
import torch
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import yaml
import glob

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

output_sum = None
num_outputs = 0
import pdb; pdb.set_trace()
model = phyla(name='phyla-beta').load().cuda()
model.eval()

print("Running inference on each concatenation of chunks...")
for i in tqdm(range(sequence_length // chunk_size)):
    with torch.no_grad():
        concats = []
        for id in id_to_sequence:
            concats.append(id_sequence_chunk[(id, i)])
        encoded_aa, cls_token_mask, sequence_mask, sequence_names = model.encode(concats[i], None)
        preds = model(encoded_aa, sequence_mask, cls_token_mask)

    if output_sum is None:
        output_sum = preds
    else:
        output_sum += preds
    num_outputs += 1

    del encoded_aa, cls_token_mask, sequence_mask, sequence_names, preds
    gc.collect()
    torch.cuda.empty_cache()

final_output = output_sum / num_outputs
final_output = final_output.cpu()
torch.save(final_output, "averaged_embeddings.pt")
print(f"Saved averaged embeddings with shape: {final_output.shape}")