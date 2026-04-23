import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load PCA coordinates from Excel file
df = pd.read_excel("pca_distances.xlsx")

# Extract cell types and coordinates
celltypes = df["Cell Type"].tolist()
gnn_pca = df[["PC1_GNN", "PC2_GNN"]].values
gpt_pca = df[["PC1_GPT", "PC2_GPT"]].values

# Set color map
colors = cm.get_cmap('tab20', len(celltypes))

plt.figure(figsize=(16, 12))

for idx, celltype in enumerate(celltypes):
    plt.scatter(gnn_pca[idx, 0], gnn_pca[idx, 1], color=colors(idx), marker='^', s=150, label=f'{celltype}: GNN')
    plt.scatter(gpt_pca[idx, 0], gpt_pca[idx, 1], color=colors(idx), marker='o', s=150, label=f'{celltype}: scGPT')

# Ensure only one label per cell type appears in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend = plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', prop={'weight': 'bold'})

# Reduce the size of the legend box
legend.get_frame().set_alpha(0.9)

plt.xlabel('First Principal Component', fontsize=20, fontweight='bold')
plt.ylabel('Second Principal Component', fontsize=20, fontweight='bold')
plt.title('PCA of Average Importance of Grad-CAMs (Updated Points)', fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.grid(True)

plt.tight_layout()  # Adjust the layout to make space for the legend
plt.savefig(f'./PCA_SAVE/PCA_Results_Updated.png', dpi=300, bbox_inches='tight')
plt.close()
