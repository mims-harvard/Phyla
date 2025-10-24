import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# --- 1. Load Embeddings ---
embedding_file = "averaged_embeddings.pt"  # <-- Replace with your file name
embeddings = torch.load(embedding_file)
if embeddings.dim() == 3:
    embeddings = embeddings.squeeze(0)  # Shape: [92, 256]

# --- 2. Load Isolate Names and Lineages ---
csv_file = "/n/holylfs06/LABS/mzitnik_lab/Lab/phyla_data_share/tb_data/metadata_circContig.csv"  # <-- Replace with your actual CSV
df = pd.read_csv(csv_file)
isolate_names = df["SampleID"].tolist()
lineages = df["PrimaryLineage_Asm"].tolist()

assert len(isolate_names) == embeddings.shape[0], "Mismatch between isolates and embeddings!"

# --- 3. Compute Pairwise Distance Matrix ---
dist_matrix = torch.cdist(embeddings, embeddings).numpy()
condensed_dist = squareform(dist_matrix)  # Convert to condensed form

# --- 4. Hierarchical Clustering (Approx. NJ) ---
Z = hierarchy.linkage(condensed_dist, method='average')  # Average ≈ NJ proxy

# --- 5. Lineage Coloring ---
le = LabelEncoder()
lineage_colors = le.fit_transform(lineages)
palette = sns.color_palette("hsv", len(le.classes_))
colors = [palette[i] for i in lineage_colors]

# --- 6. Plot the Tree ---
fig, ax = plt.subplots(figsize=(18, 12))
dendro = hierarchy.dendrogram(
    Z,
    labels=isolate_names,
    leaf_rotation=90,
    leaf_font_size=8,
    above_threshold_color='black',
    link_color_func=lambda k: 'black',
    ax=ax
)

# Recolor tip labels
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    isolate = lbl.get_text()
    idx = isolate_names.index(isolate)
    lbl.set_color(colors[idx])

plt.title("Neighbor-Joining Tree from Embeddings")
plt.tight_layout()
plt.savefig("neighbor_joining_tree.png", dpi=300)
plt.show()