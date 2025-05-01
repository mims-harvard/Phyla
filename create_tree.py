import torch
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib.patches import Patch

def get_newick(node, parent_dist, leaf_names, newick=''):
    """Convert scipy hierarchical clustering to Newick format."""
    if node.is_leaf():
        return f"{leaf_names[node.id]}:{parent_dist:.6f}{newick}"
    else:
        if len(newick) > 0:
            newick = f"):{parent_dist:.6f}{newick}"
        else:
            newick = ");"
        newick = get_newick(node.get_right(), node.dist/2.0, leaf_names, newick)
        newick = get_newick(node.get_left(), node.dist/2.0, leaf_names, f",{newick}")
        newick = f"({newick}"
        return newick

# Load embeddings
embeddings = torch.load('averaged_embeddings.pt')
if embeddings.dim() == 3:
    embeddings = embeddings.squeeze(0)

# Load metadata
metadata = pd.read_csv('metadata_circContig.csv')

# Calculate pairwise distances
distances = torch.cdist(embeddings, embeddings).numpy()

# Create linkage matrix for neighbor joining
Z = linkage(distances, method='average')

# Create color mapping for lineages
unique_lineages = metadata['PrimaryLineage_Asm'].unique()
colors = sns.color_palette('husl', n_colors=len(unique_lineages))
lineage_to_color = dict(zip(unique_lineages, colors))

# Create labels with sample IDs
labels = metadata['SampleID'].values

# Create the dendrogram
plt.figure(figsize=(15, 10))
dend = dendrogram(
    Z,
    labels=labels,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=0.7 * max(Z[:, 2])
)

# Color the leaves based on lineage
ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    sample_id = lbl.get_text()
    lineage = metadata[metadata['SampleID'] == sample_id]['PrimaryLineage_Asm'].iloc[0]
    lbl.set_color(lineage_to_color[lineage])

# Create legend
legend_elements = [Patch(facecolor=color, label=lineage)
                  for lineage, color in lineage_to_color.items()]
plt.legend(handles=legend_elements, title='Lineages', 
          bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Neighbor Joining Tree of Isolates')
plt.ylabel('Average Pairwise Distance')
plt.tight_layout()
plt.savefig('neighbor_joining_tree.png', dpi=300, bbox_inches='tight')
plt.close() 


tree = to_tree(Z, False)
newick = get_newick(tree, tree.dist, labels)

# Save Newick tree to file
with open('tree.newick', 'w') as f:
    f.write(newick)