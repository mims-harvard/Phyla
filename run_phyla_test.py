from phyla import phyla

seq = "./phyla/data/40seqs.fasta"
model = phyla().load().cuda()
encoded_aa, cls_token_mask, sequence_mask, sequence_names = model.encode_fasta(seq)
preds = model(encoded_aa, sequence_mask, cls_token_mask)
tree = model.reconstruct_tree(preds, sequence_names)
print(tree)
