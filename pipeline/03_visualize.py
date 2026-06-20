"""
03_visualize.py
===============
Visualise clustering results after running 01 and 02.

Produces three figure files in pipeline/out/:
  fig1_clustering.png  — 3D PCA scatter + cluster sizes + feature box-plots
  fig2_timeseries.png  — cluster label + key features over time
  fig3_correlation.png — feature correlation heatmap

Run:
    python pipeline/03_visualize.py
"""

import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT    = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pipeline" / "out"

PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

FEATURE_NAMES = [
    "mean", "std", "ptp", "slope",
    "zcr", "spike_count",
    "hjorth_mobility", "hjorth_complexity", "rms_first_diff",
]


# ── Load artefacts ────────────────────────────────────────────────────────────
def load_all():
    model_path = OUT_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run 02_train.py first."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df     = pd.read_csv(OUT_DIR / "features_clustered.csv")
    pca_df = pd.read_csv(OUT_DIR / "pca_coords.csv")
    return model, df, pca_df


# ── Figure 1: 3D scatter + cluster sizes + feature box-plots ─────────────────
def fig1(model, df, pca_df) -> None:
    km     = model["kmeans"]
    labels = model["labels"]
    K      = km.n_clusters

    fig = plt.figure(figsize=(18, 11))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    # 3D scatter
    ax3 = fig.add_subplot(gs[0, :2], projection="3d")
    for k in range(K):
        m = labels == k
        ax3.scatter(
            pca_df.loc[m, "pc1"], pca_df.loc[m, "pc2"], pca_df.loc[m, "pc3"],
            c=PALETTE[k % len(PALETTE)], label=f"Cluster {k}",
            s=18, alpha=0.65, linewidths=0,
        )
    ctrs = km.cluster_centers_
    ax3.scatter(ctrs[:, 0], ctrs[:, 1], ctrs[:, 2],
                c="black", marker="X", s=160, zorder=6, label="Centroids")
    ax3.set_xlabel("PC 1", labelpad=6)
    ax3.set_ylabel("PC 2", labelpad=6)
    ax3.set_zlabel("PC 3", labelpad=6)
    ax3.set_title(f"K-means in PCA space  (k = {K})", fontsize=13, pad=12)
    ax3.legend(fontsize=8, loc="upper left")

    # Cluster bar chart
    ax_bar = fig.add_subplot(gs[0, 2])
    counts = np.bincount(labels, minlength=K)
    bars   = ax_bar.bar(range(K), counts,
                        color=[PALETTE[k % len(PALETTE)] for k in range(K)],
                        width=0.6, edgecolor="white")
    ax_bar.set_xlabel("Cluster")
    ax_bar.set_ylabel("Windows")
    ax_bar.set_title("Cluster sizes")
    ax_bar.set_xticks(range(K))
    for k, (bar, c) in enumerate(zip(bars, counts)):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, c + 0.5,
                    str(c), ha="center", va="bottom", fontsize=9)

    # Feature box-plots per cluster
    ax_feat = fig.add_subplot(gs[1, :])
    n_feat  = len(FEATURE_NAMES)
    spacing = K + 1

    for k in range(K):
        mask   = labels == k
        data_k = [df.loc[mask, f].to_numpy() for f in FEATURE_NAMES]
        pos    = [k * 0.75 + i * spacing for i in range(n_feat)]
        ax_feat.boxplot(
            data_k, positions=pos, widths=0.55, patch_artist=True,
            boxprops=dict(facecolor=PALETTE[k % len(PALETTE)], alpha=0.55),
            medianprops=dict(color="black", linewidth=1.8),
            whiskerprops=dict(color=PALETTE[k % len(PALETTE)]),
            capprops=dict(color=PALETTE[k % len(PALETTE)]),
            flierprops=dict(marker=".", color=PALETTE[k % len(PALETTE)],
                            markersize=2, alpha=0.35),
            showfliers=False,
        )

    tick_pos = [
        np.mean([k * 0.75 + i * spacing for k in range(K)])
        for i in range(n_feat)
    ]
    ax_feat.set_xticks(tick_pos)
    ax_feat.set_xticklabels(FEATURE_NAMES, rotation=32, ha="right", fontsize=10)
    ax_feat.set_title("Feature distributions per cluster")
    ax_feat.set_ylabel("Raw feature value")
    legend_elems = [
        Patch(facecolor=PALETTE[k % len(PALETTE)], label=f"Cluster {k}")
        for k in range(K)
    ]
    ax_feat.legend(handles=legend_elems, loc="upper right", fontsize=9)

    out = OUT_DIR / "fig1_clustering.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


# ── Figure 2: cluster label + key features over time ─────────────────────────
def fig2(model, df, pca_df) -> None:
    labels = model["labels"]
    K      = model["kmeans"].n_clusters
    t      = np.arange(len(labels)) * (50 / 100.0)  # HOP / FS seconds per window

    tracked_feats = ["mean", "spike_count", "hjorth_complexity"]
    fig, axes = plt.subplots(1 + len(tracked_feats), 1,
                             figsize=(15, 9), sharex=True)

    # Cluster label scatter
    axes[0].scatter(t, labels,
                    c=[PALETTE[l % len(PALETTE)] for l in labels],
                    s=8, linewidths=0)
    axes[0].set_ylabel("Cluster")
    axes[0].set_yticks(range(K))
    axes[0].set_title("Cluster label & key features over time")

    # Feature time-series with cluster shading
    for ax, feat in zip(axes[1:], tracked_feats):
        ax.plot(t, df[feat].to_numpy(), lw=0.75, color="#2c3e50", alpha=0.85)
        ax.set_ylabel(feat, fontsize=9)
        # shade spans by cluster
        for k in range(K):
            idxs = np.where(labels == k)[0]
            for idx in idxs:
                ax.axvspan(t[idx] - 0.25, t[idx] + 0.25,
                           color=PALETTE[k % len(PALETTE)], alpha=0.08,
                           linewidth=0)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    out = OUT_DIR / "fig2_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


# ── Figure 3: feature correlation heatmap ────────────────────────────────────
def fig3(df) -> None:
    feat_mat = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    corr     = np.corrcoef(feat_mat.T)
    n        = len(FEATURE_NAMES)

    fig, ax = plt.subplots(figsize=(10, 8))
    im      = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.82)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(FEATURE_NAMES, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_title("Feature correlation matrix")
    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr[i, j]) > 0.65 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    plt.tight_layout()
    out = OUT_DIR / "fig3_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    model, df, pca_df = load_all()

    print(f"Loaded model: k={model['kmeans'].n_clusters} clusters, "
          f"{len(model['labels'])} windows")

    fig1(model, df, pca_df)
    fig2(model, df, pca_df)
    fig3(df)

    plt.show()


if __name__ == "__main__":
    main()
