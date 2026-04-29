from pathlib import Path
import pickle
import re
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

'''
Used for visualization purposes: Scatter plots for predicted and annotated values
'''


dataset_name = "ms"

##################################################################################
if dataset_name == "ms":
    data_dir = Path("./data/ms")
    adata = sc.read(data_dir / "c_data.h5ad")
    adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
    adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test.var.set_index(adata.var["gene_name"], inplace=True)
    data_is_raw = False
    filter_gene_by_counts = False
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"] = np.arange(adata.obs.shape[0])

if dataset_name == "pancreas":
    data_dir = Path("./data/pancreas")
    adata = sc.read(data_dir / "demo_train.h5ad")
    adata_test = sc.read(data_dir / "demo_test.h5ad")
    adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
    data_is_raw = False
    filter_gene_by_counts = False
    adata_test_raw = adata_test.copy()
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"] = np.arange(adata.obs.shape[0])

if dataset_name == "myeloid":
    data_dir = Path("./data/mye/")
    adata = sc.read(data_dir / "reference_adata.h5ad")
    adata_test = sc.read(data_dir / "query_adata.h5ad")
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
    adata_test_raw = adata_test.copy()
    data_is_raw = False
    filter_gene_by_counts = False
    adata = adata.concatenate(adata_test, batch_key="str_batch")
    adata.obs["indices"] = np.arange(adata.obs.shape[0])
##################################################################################


# Take results from the saved transformer model
file_path = os.path.join(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{dataset_name}_median/results.pkl")
with open(file_path, "rb") as file:
    results = pickle.load(file)
seed_list = results["seed_numbers"]


path_to_plot = "/auto/k2/aykut3/Yunus/scarce/scarce_merged/ms/type4/GC-CG/portion_5/dname_ms_path_[GC-CG]_type_type4_seedid_2_seed_31_portion_0.05"

with open(path_to_plot, "rb") as f:
    loaded_results = pickle.load(f)

y_test_preds = loaded_results["test_preds"][-1]


# Map prediction integer ids to human-readable cell type names
id2type = results["id_maps"]
print(id2type)


def _clean_label(s):
    """Strip '?' characters and collapse extra whitespace from a cell-type name."""
    return re.sub(r"\s+", " ", str(s).replace("?", "")).strip()


id2type = {k: _clean_label(v) for k, v in id2type.items()}
adata_test_raw.obs["predictions"] = [id2type[p] for p in y_test_preds]
adata_test_raw.obs["celltype"] = adata_test_raw.obs["celltype"].astype(str).map(_clean_label)

all_categories = sorted(
    set(adata_test_raw.obs["celltype"].astype(str).unique())
    | set(adata_test_raw.obs["predictions"].astype(str).unique())
)
for _col in ("celltype", "predictions"):
    adata_test_raw.obs[_col] = (
        adata_test_raw.obs[_col].astype(str).astype("category")
        .cat.set_categories(all_categories)
    )

try:
    from scanpy.plotting import palettes as _sc_palettes
    if len(all_categories) <= 20:
        _base_colors = list(_sc_palettes.default_20)
    elif len(all_categories) <= 28:
        _base_colors = list(_sc_palettes.default_28)
    else:
        _base_colors = list(_sc_palettes.default_102)
except Exception:
    _base_colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)

palette_ = {c: _base_colors[i % len(_base_colors)] for i, c in enumerate(all_categories)}


# --------------------------------------------------------------------------- #
# Publication-quality side-by-side UMAP: ground truth vs. predicted cell type
# --------------------------------------------------------------------------- #
n_cells = adata_test_raw.n_obs
# Bigger, more visible markers
marker_size = max(70, min(180, 400000 / max(n_cells, 1)))

_portion_match = re.search(r"portion_(\d+)", path_to_plot)
portion_pct = int(_portion_match.group(1)) if _portion_match else 5

rc = {
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 1.2,
    "legend.frameon": False,
}

with plt.rc_context(rc):
    # Stacked layout (single column): two large panels, one on top of the other.
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 18.0), constrained_layout=False)

    common = dict(
        palette=palette_,
        show=False,
        size=marker_size,
        frameon=True,
        legend_loc=None,
        na_color="lightgray",
    )
    sc.pl.umap(adata_test_raw, color="celltype",    ax=axes[0], title="", **common)
    sc.pl.umap(adata_test_raw, color="predictions", ax=axes[1], title="", **common)

    panel_titles = ("Ground Truth", "Predicted")
    for ax, t in zip(axes, panel_titles):
        ax.set_title(t, fontsize=16, fontweight="bold", color="black", pad=8)
        ax.set_xlabel("UMAP 1", fontsize=14, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(3.0)
            ax.spines[side].set_color("black")
            ax.spines[side].set_zorder(10)

    fig.suptitle(
        f"Test Results for Training with {portion_pct}% Labeled Samples",
        fontsize=18, fontweight="bold", y=0.985,
    )

    # Larger legend markers and readable text
    handles = [mpatches.Patch(facecolor=palette_[c], edgecolor="none", label=c)
               for c in all_categories]
    n_legend_cols = 2
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=n_legend_cols,
        fontsize=28,
        title="Cell Type",
        title_fontsize=30,
        handlelength=2.0,
        handletextpad=1.0,
        columnspacing=2.0,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_title().set_fontweight("bold")

    plt.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.10, hspace=0.18)

    out_png = f"results_combined_{dataset_name}.png"
    out_pdf = f"results_combined_{dataset_name}.pdf"
    plt.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

plt.show()