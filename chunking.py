from phyla import phyla
import torch
from tqdm import tqdm
import os
import gc

num_files = 1000
sequence_length = 1_000_000
chunk_size = 500
output_sum = None
num_outputs = 0

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = phyla(name='phyla-alpha').load().cuda()
model.eval()

input_dir = "/n/home10/ljain/Phyla/data/fasta_files"
concats = [[] for _ in range(sequence_length // chunk_size)] # the ith element in this list is a list of the ith chunks from all files

print(f"Reading files in and loading lists...")
file_count = 0
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".fasta"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as file:
            for line in file:
                if line[0] == ">":
                    continue
                else:
                    for i in range(sequence_length // chunk_size):
                        concats[i].append(line.strip()[i * chunk_size:(i + 1) * chunk_size])
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

    del encoded_aa, cls_token_mask, sequence_mask, sequence_names, preds
    gc.collect()
    torch.cuda.empty_cache()


final_output = output_sum / num_outputs
print(final_output.shape)