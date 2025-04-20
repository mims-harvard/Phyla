from phyla import phyla
import torch
from tqdm import tqdm
import os
import gc

num_files = 92
sequence_length = 4_500_000
chunk_size = 500
output_sum = None
num_outputs = 0

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = phyla(name='phyla-alpha').load().cuda()
model.eval()

input_dir = "/n/holylfs06/LABS/mzitnik_lab/Lab/phyla_data_share/tb_data/assemblies" # there are 92 fasta files in this directory
concats = [[] for _ in range(sequence_length // chunk_size)] # the ith element in this list is a list of the ith chunks from all files

print(f"Reading files in and loading lists...")
file_count = 0
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".fasta"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()
        
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
        for i in range(sequence_length // chunk_size):
            string_to_add = sequence[i * chunk_size:(i + 1) * chunk_size].ljust(chunk_size)
            concats[i].append(string_to_add)
        
        file_count += 1
        if file_count >= num_files:
            break

print("Running inference on each concatenation of chunks...")
for i in tqdm(range(sequence_length // chunk_size)):
    with torch.no_grad():
        encoded_aa, cls_token_mask, sequence_mask, sequence_names = model.encode(concats[i], None)
        preds = model(encoded_aa, sequence_mask, cls_token_mask)

    if output_sum is None:
        output_sum = preds
    else:
        output_sum += preds
    num_outputs += 1

    # if 'all_outputs' not in globals():
    #     all_outputs = []
    # all_outputs.append(preds.cpu())

    del encoded_aa, cls_token_mask, sequence_mask, sequence_names, preds
    gc.collect()
    torch.cuda.empty_cache()


# final_embeddings = torch.cat(all_outputs, dim=0) 

# # Save to file
# torch.save(final_embeddings, "embeddings.pt")
# print(f"Saved embeddings with shape: {final_embeddings.shape}")

final_output = output_sum / num_outputs
final_output = final_output.cpu()
torch.save(final_output, "averaged_embeddings.pt")
print(f"Saved averaged embeddings with shape: {final_output.shape}")