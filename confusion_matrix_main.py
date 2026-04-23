from pathlib import Path
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


"""
In the paper, the figures obtained from this code is utilized, in supp material
"""

dataset_name="myeloid"

##################################################################################
if dataset_name == "ms":
    data_dir = Path("../data/ms")
    adata = sc.read(data_dir / "c_data.h5ad")
    adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
    adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test.var.set_index(adata.var["gene_name"], inplace=True)
    data_is_raw = False
    filter_gene_by_counts = False
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"]= np.arange(adata.obs.shape[0])
    path_to_plot = "/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/scgnn_merged/ms/type3/GC-CG/dname_ms_path_[GC-CG]_type_type3_seedid_3_seed_35"
    class_names = ['PVALB-expressing interneuron', 'SST-expressing interneuron', 'SV2C-expressing interneuron', 'VIP-expressing interneuron', 'astrocyte', 'cortical layer 2-3 excitatory neuron A', 'cortical layer 2-3 excitatory neuron B' ,'cortical layer 4 excitatory neuron',
 'cortical layer 5-6 excitatory neuron', 'endothelial cell',
 'microglial cell', 'mixed excitatory neuron', 'mixed glial cell?',
 'oligodendrocyte A', 'oligodendrocyte precursor cell', 'pyramidal neuron?', 'oligodendrocyte C',
 'phagocyte']

if dataset_name == "pancreas": #RB
    data_dir = Path("../data/pancreas")
    adata = sc.read(data_dir / "demo_train.h5ad")
    adata_test = sc.read(data_dir / "demo_test.h5ad")
    adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"    
    data_is_raw = False
    filter_gene_by_counts = False   
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"]= np.arange(adata.obs.shape[0])
    path_to_plot = "/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/scgnn_merged/pancreas/type3/GC-CG/dname_pancreas_path_[GC-CG]_type_type3_seedid_6_seed_24"
    class_names = ['PP', 'PSC', 'acinar', 'alpha', 'beta', 'delta', 'ductal', 'endothelial', 'epsilon', 'mast', 'MHC class II','macrophage']

if dataset_name == "myeloid":
    data_dir = Path("../data/mye/")
    adata = sc.read(data_dir / "reference_adata.h5ad")
    adata_test = sc.read(data_dir / "query_adata.h5ad")
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
    adata_test_raw = adata_test.copy()
    data_is_raw = False
    filter_gene_by_counts = False   
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"]= np.arange(adata.obs.shape[0])
    path_to_plot = "/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/scgnn_merged/myeloid/type3/GC-CG/dname_myeloid_path_[GC-CG]_type_type3_seedid_3_seed_9"
    class_names = ['Macro_C1QC', 'Macro_INHBA', 'Macro_LYVE1', 'Macro_NLRP3', 'Macro_SPP1', 'Mono_CD14', 'Mono_CD16', 'cDC1_CLEC9A', 'cDC2_CD1C', 'cDC3_LAMP3', 'pDC_LILRA4',
    'Macro_FN1', 'Macro_GPNMB', 'Macro_IL1B', 'Macro_ISG15', 'cDC2_CD1A', 'cDC2_CXCR4hi',
   'cDC2_FCN1', 'cDC2_IL1B', 'cDC2_ISG15']

##################################################################################
from matplotlib.cm import get_cmap
file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/results.pkl")
with open(file_path, "rb") as file:
    results = pickle.load(file)

seed_list = results["seed_numbers"]

with open(path_to_plot, "rb") as f:
    loaded_results = pickle.load(f)

y_test_preds = loaded_results["test_preds"][-1] # scgrapgt predictions
y_test_preds = results["predictions"] ### Don't comment if you would like to use scgpt reproducted predictions



# Map predictions to cell type labels
id2type = results["id_maps"]
adata_test_raw.obs["predictions"] = [id2type[p] for p in y_test_preds]

# Get true and predicted labels
y_true = adata_test_raw.obs["celltype"]
y_pred = adata_test_raw.obs["predictions"]

classes_to_combine = [
    'Macro_FN1', 'Macro_GPNMB', 'Macro_IL1B', 'Macro_ISG15', 
    'cDC2_CD1A', 'cDC2_CXCR4hi', 'cDC2_FCN1', 'cDC2_IL1B', 'cDC2_ISG15'
]

# Function to map classes to "others"
def map_to_others(class_label):
    if class_label in classes_to_combine:
        return "others"
    return class_label

if dataset_name == "myeloid":
    y_true = y_true.map(map_to_others)
    y_pred = y_pred.map(map_to_others)
    original_class_names = class_names
    class_names = [cls for cls in original_class_names if cls not in classes_to_combine]
    class_names.append('others')

# Print unique values in y_true and y_pred
print("Unique values in y_true:", np.unique(y_true))
print("Unique values in y_pred:", np.unique(y_pred))

# Explicitly set the order of class names to match the desired order in the image


# Get colors from the tab20 colormap
tab20 = get_cmap('tab20')
class_colors = [tab20(i/len(class_names)) for i in range(len(class_names))]

# Convert y_true and y_pred to category codes using the specified class order
y_true_codes = pd.Categorical(y_true, categories=class_names).codes
y_pred_codes = pd.Categorical(y_pred, categories=class_names).codes

def plot_confusion_matrix(y_true, y_pred, class_names, class_colors):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Px for Myeloid
    """
    width_px = 660
    height_px = 720
    """
    
    width_px = 660
    height_px = 660
        
    # Set DPI (dots per inch)
    dpi = 100  # For example, 100 pixels per inch
    
    # Calculate figsize in inches
    figsize = (width_px / dpi, height_px / dpi)
    
    # Create the figure with the specified size and DPI
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Plot the heatmap without tick labels and color bar
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Purples', 
                xticklabels=False, yticklabels=False, cbar=False, 
                linewidths=0, annot_kws={"size": 10})

    plt.xlabel('')
    plt.ylabel('')
    #plt.title('Confusion Matrix (MS)', fontsize=20)
    
    # Remove the axis ticks
    plt.tick_params(left=False, bottom=False)
    
    plt.tight_layout()
    plt.savefig('Mye_Graph_CM.png', dpi=dpi, bbox_inches='tight', pad_inches=0) # MS_GPT_CM.png
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(y_true_codes, y_pred_codes, class_names, class_colors)