"""
Generate publication-quality reproductions of all figures in the paper:
  - Figure 3: Mutual information analysis of four feature types (2, 3, 4, 5 tabs)
  - Figure 7: Precision-recall curves in the closed-world scenario (2, 3, 4, 5 tabs)
  - Figure 8: Open-world AUC_M and AUC_N across tab settings (2, 3, 4, 5 tabs)
  - Figure 9: Parameter sensitivity analysis of ARES (t, d, m, n)
  - Figure 10: Training and inference latency vs. number of monitored websites
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    }
)


def plot_fig3_mutual_information(mi_data, out_dir):
    tabs = [2, 3, 4, 5]
    x = np.arange(len(tabs))
    width = 0.18

    feat_names = [
        "Packet sequence",
        "Packet-level aggregation",
        "Burst-level aggregation",
        "Multi-level aggregation",
    ]
    colors = ["#4a86c5", "#e68a4d", "#459e68", "#d64541"]
    hatches = ["///", "...", "\\\\\\", "**"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for idx, fname in enumerate(feat_names):
        vals = [mi_data[str(t)][fname] if str(t) in mi_data else mi_data[t][fname] for t in tabs]
        ax.bar(
            x + (idx - 1.5) * width,
            vals,
            width,
            label=fname,
            facecolor="white",
            edgecolor=colors[idx],
            hatch=hatches[idx],
            linewidth=1.5,
        )

    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("Number of tabs")
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in tabs])
    ax.set_ylim(0, 4.6)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig3_mutual_information.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig7_pr_curves(pr_data, out_dir):
    models_order = ["ARES", "TMWF", "Var-CNN", "Tik-Tok", "RF", "DF", "BAPM", "NetCLR"]
    styles = {
        "ARES": {"color": "#d62728", "linestyle": "-", "linewidth": 2.2},
        "TMWF": {"color": "#e377c2", "linestyle": "--", "linewidth": 1.8},
        "Var-CNN": {"color": "#8c564b", "linestyle": ":", "linewidth": 1.8},
        "Tik-Tok": {"color": "#ffbb78", "linestyle": "-.", "linewidth": 1.8},
        "RF": {"color": "#7f7f7f", "linestyle": "-", "linewidth": 1.6},
        "DF": {"color": "#2ca02c", "linestyle": "--", "linewidth": 1.8},
        "BAPM": {"color": "#9467bd", "linestyle": ":", "linewidth": 1.8},
        "NetCLR": {"color": "#1f77b4", "linestyle": "-.", "linewidth": 1.8},
    }

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6), sharey=False)
    tab_labels = ["(a) 2-tab setting", "(b) 3-tab setting", "(c) 4-tab setting", "(d) 5-tab setting"]

    for idx, k_tab in enumerate([2, 3, 4, 5]):
        ax = axes[idx]
        key = str(k_tab) if str(k_tab) in pr_data else k_tab
        for m in models_order:
            if m in pr_data[key]:
                rec = pr_data[key][m]["recall"]
                prec = pr_data[key][m]["precision"]
                ax.plot(rec, prec, label=m, **styles[m])
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel(f"Recall\n{tab_labels[idx]}")
        if idx == 0:
            ax.set_ylabel("Precision")
        ax.grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.08), frameon=False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig7_pr_curves_closed_world.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig8_open_world_auc(ow_auc_data, out_dir):
    models_order = ["ARES", "TMWF", "Var-CNN", "Tik-Tok", "RF", "DF", "BAPM", "NetCLR"]
    colors = ["#d64541", "#b07aa1", "#e68a4d", "#d4a64f", "#59a14f", "#4a86c5", "#8c564b", "#7f7f7f"]
    hatches = ["", "\\\\\\", "///", "---", "...", "\\\\", "//", ""]

    tabs = [2, 3, 4, 5]
    x = np.arange(len(tabs))
    width = 0.095

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True)

    for idx, m in enumerate(models_order):
        aucm_vals = [
            ow_auc_data[str(t) if str(t) in ow_auc_data else t][m]["AUC_M"] for t in tabs
        ]
        aucn_vals = [
            ow_auc_data[str(t) if str(t) in ow_auc_data else t][m]["AUC_N"] for t in tabs
        ]
        offset = (idx - 3.5) * width
        ax1.bar(
            x + offset,
            aucm_vals,
            width,
            label=m,
            facecolor="white",
            edgecolor=colors[idx],
            hatch=hatches[idx],
            linewidth=1.3,
        )
        ax2.bar(
            x + offset,
            aucn_vals,
            width,
            label=m,
            facecolor="white",
            edgecolor=colors[idx],
            hatch=hatches[idx],
            linewidth=1.3,
        )

    ax1.set_ylabel(r"$\mathrm{AUC}_M$")
    ax1.set_ylim(0.6, 1.02)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.set_ylabel(r"$\mathrm{AUC}_N$")
    ax2.set_xlabel("Number of tabs")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in tabs])
    ax2.set_ylim(0.6, 1.02)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), frameon=False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig8_open_world_auc.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig9_parameter_analysis(param_data, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))

    configs = [
        ("t", ["10", "20", "30", "40", "50"], "Value of $t$\n(a) Time interval", "#d98282", "#a83232"),
        ("d", ["2k", "4k", "6k", "8k", "10k"], "Value of $d$\n(b) Input dimension", "#e8b382", "#b86b23"),
        ("m", ["5", "10", "15", "20", "25"], "Value of $m$\n(c) Top-m attention", "#b5d69c", "#4d8028"),
        ("n", ["1", "2", "3", "4", "5"], "Value of $n$\n(d) # of attention layers", "#9ccfb5", "#287855"),
    ]

    for idx, (key, xlabels, xlabel_text, fcolor, ecolor) in enumerate(configs):
        ax = axes[idx]
        vals = param_data[key]
        x = np.arange(len(xlabels))
        ax.bar(x, vals, width=0.55, facecolor=fcolor, edgecolor=ecolor, linewidth=1.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_ylim(0.6, 1.0)
        ax.set_xlabel(xlabel_text)
        if idx == 0:
            ax.set_ylabel("MAP@2")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig9_parameter_analysis.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig10_latency_analysis(latency_data, out_dir):
    sites = [20, 40, 60, 80, 100]
    x = np.arange(len(sites))
    width = 0.34

    train_lat = [latency_data[str(s) if str(s) in latency_data else s]["train_ms"] for s in sites]
    infer_lat = [latency_data[str(s) if str(s) in latency_data else s]["infer_ms"] for s in sites]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    rects1 = ax.bar(
        x - width / 2,
        train_lat,
        width,
        label="Training latency",
        facecolor="white",
        edgecolor="#3d65a5",
        hatch="///",
        linewidth=1.5,
    )
    rects2 = ax.bar(
        x + width / 2,
        infer_lat,
        width,
        label="Inference latency",
        facecolor="white",
        edgecolor="#c94a38",
        hatch="\\\\\\",
        linewidth=1.5,
    )

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Number of monitored websites")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sites])
    ax.set_ylim(0, max(30.0, max(train_lat) * 1.2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig10_latency_analysis.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_figures(results_dict, out_dir="./results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    if "fig3_mutual_info" in results_dict:
        paths["fig3"] = plot_fig3_mutual_information(results_dict["fig3_mutual_info"], out_dir)
    if "fig7_pr_curves" in results_dict:
        paths["fig7"] = plot_fig7_pr_curves(results_dict["fig7_pr_curves"], out_dir)
    if "fig8_open_world_auc" in results_dict:
        paths["fig8"] = plot_fig8_open_world_auc(results_dict["fig8_open_world_auc"], out_dir)
    if "fig9_parameter_analysis" in results_dict:
        paths["fig9"] = plot_fig9_parameter_analysis(results_dict["fig9_parameter_analysis"], out_dir)
    if "fig10_latency" in results_dict:
        paths["fig10"] = plot_fig10_latency_analysis(results_dict["fig10_latency"], out_dir)
    return paths
