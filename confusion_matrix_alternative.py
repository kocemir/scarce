from pathlib import Path
import pickle
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


dataset_name="pancreas"

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
##################################################################################


#### Take results from the save transformer model
file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/results.pkl")
with open(file_path, "rb") as file:
        results= pickle.load(file)   
seed_list=results["seed_numbers"]


# We can automatize this, but I just want some visual examples
path_to_plot= "/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/scgnn_merged/pancreas/type3/GC-CG/dname_pancreas_path_[GC-CG]_type_type3_seedid_6_seed_24"

with open(path_to_plot, "rb") as f:
    loaded_results = pickle.load(f)

y_test_preds= loaded_results["test_preds"][-1]
y_test_preds= results["predictions"]
id2type=results["id_maps"]
adata_test_raw.obs["predictions"]=[id2type[p]  for p in  y_test_preds]

# Assuming adata_test_raw.obs["predictions"] and adata_test_raw.obs["target"] are pandas Series
predictions = adata_test_raw.obs["predictions"]
target = adata_test_raw.obs["celltype"]

# Create a union of all unique labels in both predictions and target
all_labels = np.union1d(predictions.unique(), target.unique())

# Generate the confusion matrix using the union of labels
cm = confusion_matrix(target, predictions, labels=all_labels, normalize='true')
cm = np.round(cm, 2)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
disp.plot(cmap='viridis', xticks_rotation='vertical')

# Save the plot as an image file (e.g., PNG)
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')