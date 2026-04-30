import os
import re
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# --- Configuration ---
DATASET_NAME = "ms"
SAVE_PATH = Path(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{DATASET_NAME}_median/results.pkl")
PLOT_DATA_PATH = Path("/auto/k2/aykut3/Yunus/scarce/scarce_merged/ms/type4/GC-CG/portion_5/dname_ms_path_[GC-CG]_type_type4_seedid_2_seed_31_portion_0.05")

def clean_label(s):
    return re.sub(r"\s+", " ", str(s).replace("?", "")).strip()

def load_and_preprocess_data(dataset):
    data_dir = Path(f"./data/{dataset}")
    if dataset == "ms":
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        obs_col = "Factor Value[inferred cell type - authors labels]"
    elif dataset == "pancreas":
        adata = sc.read(data_dir / "demo_train.h5ad")
        adata_test = sc.read(data_dir / "demo_test.h5ad")
        obs_col = "Celltype"
    else:
        adata_test = sc.read(data_dir / "query_adata.h5ad")
        obs_col = "cell_type"

    adata_test.obs["celltype"] = adata_test.obs[obs_col].astype(str).apply(clean_label)
    return adata_test.copy()

def plot_results(adata_viz, results_dict, plot_results_dict, dataset_name):
    # 1. Map Predictions
    id2type = {k: clean_label(v) for k, v in results_dict["id_maps"].items()}
    y_preds = plot_results_dict["test_preds"][-1]
    adata_viz.obs["predictions"] = [id2type[p] for p in y_preds]

    # 2. Categories and Palette
    all_cats = sorted(set(adata_viz.obs["celltype"]) | set(adata_viz.obs["predictions"]))
    adata_viz.obs["celltype"] = pd.Categorical(adata_viz.obs["celltype"], categories=all_cats)
    adata_viz.obs["predictions"] = pd.Categorical(adata_viz.obs["predictions"], categories=all_cats)
    
    n_cats = len(all_cats)
    if n_cats <= 10:
        palette = list(plt.get_cmap("tab10").colors)
    elif n_cats <= 20:
        palette = list(plt.get_cmap("tab20").colors)
    else:
        palette = (
            list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors)
            + list(plt.get_cmap("Set3").colors)
            + list(plt.get_cmap("Set2").colors)
        )
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cats)}

    # 3. Setup Figure RC
    rc_params = {
        "figure.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 4.0,  # Thick borders like image_e3f09d.jpg
    }

    with plt.rc_context(rc_params):
        # Larger overall canvas while preserving page-friendly proportions.
        fig, axes = plt.subplots(2, 1, figsize=(13.0, 17.0))

        n_cells = adata_viz.n_obs
        size = max(50, min(150, 300000 / max(n_cells, 1)))

        plot_args = dict(palette=color_map, show=False, size=size, frameon=True, legend_loc=None)

        sc.pl.umap(adata_viz, color="celltype", ax=axes[0], title="", **plot_args)
        sc.pl.umap(adata_viz, color="predictions", ax=axes[1], title="", **plot_args)

        # 4. Axis labels and styling (sized for a single page).
        titles = ["Ground Truth", "Predicted"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=24, fontweight="bold", pad=14)
            ax.set_xlabel("UMAP 1", fontsize=20, fontweight="bold", labelpad=10)
            ax.set_ylabel("UMAP 2", fontsize=20, fontweight="bold", labelpad=10)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
                spine.set_edgecolor("black")

        # 5. Global title.
        portion_match = re.search(r"portion_(\d+)(?!\.)", str(PLOT_DATA_PATH))
        pct = f"{int(portion_match.group(1))}%" if portion_match else "5%"
        fig.suptitle(
            f"Test Results for Training with {pct} Labeled Samples",
            fontsize=28, fontweight="bold", y=0.985,
        )

        # 6. Legend below the panels with a clear buffer.
        handles = [mpatches.Patch(color=color_map[c], label=c) for c in all_cats]

        legend = fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.15),
            ncol=2,
            fontsize=19,
            title="Cell Type",
            title_fontsize=21,
            handlelength=0.72,
            handleheight=0.42,
            handletextpad=0.38,
            columnspacing=0.95,
            frameon=False,
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")
        legend.get_title().set_fontweight("bold")

        # Layout: leave room for suptitle on top and legend on the bottom.
        plt.subplots_adjust(top=0.93, bottom=0.20, hspace=0.22, left=0.08, right=0.98)

        out_name = f"final_expanded_{dataset_name}.png"
        plt.savefig(out_name, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Success! Figure saved to {out_name}")
        plt.show()

if __name__ == "__main__":
    with open(SAVE_PATH, "rb") as f: meta_results = pickle.load(f)
    with open(PLOT_DATA_PATH, "rb") as f: prediction_results = pickle.load(f)
    test_adata = load_and_preprocess_data(DATASET_NAME)
    plot_results(test_adata, meta_results, prediction_results, DATASET_NAME)