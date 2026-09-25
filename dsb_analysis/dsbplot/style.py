import matplotlib as mpl

PATH_ORDER = ["gemm-fp16", "gemm-int8", "gemm-tf32", "gemm-fp32", "bit", "block", "block-fp16",
              "cluster", "cluster-fp16", "global-sync", "global-sync-fp16",
              "csr-row", "csr-block", "csr-cluster", "auto", "public-matched", "public-library"]

NAME = {
    "gemm-fp16": "gemm FP16", "gemm-int8": "gemm INT8", "gemm-tf32": "gemm TF32",
    "gemm-fp32": "gemm FP32 (no TF32)", "bit": "bit", "block": "block (fused, R=1)",
    "block-fp16": "block FP16", "cluster": "cluster", "cluster-fp16": "cluster FP16",
    "global-sync": "global-sync", "global-sync-fp16": "global-sync FP16",
    "csr-row": "csr-row", "csr-block": "csr-block", "csr-cluster": "csr-cluster", "auto": "auto",
    "public-matched": "SB 2.0.0 (matched)", "public-library": "SB 2.0.0 (library defaults)",
}

COLOR = {
    "gemm-fp16": "#d62728", "gemm-int8": "#ff7f0e", "gemm-tf32": "#1f77b4", "gemm-fp32": "#17becf",
    "bit": "#2ca02c", "block": "#9467bd", "block-fp16": "#c5b0d5",
    "cluster": "#8c564b", "cluster-fp16": "#c49c94", "global-sync": "#7f7f7f", "global-sync-fp16": "#c7c7c7",
    "csr-row": "#8c6d31", "csr-block": "#e377c2", "csr-cluster": "#bcbd22", "auto": "#000000",
    "public-matched": "#000000", "public-library": "#555555",
}

MARKER = {
    "gemm-fp16": "o", "gemm-int8": "s", "gemm-tf32": "^", "gemm-fp32": "v", "bit": "D", "block": "P",
    "block-fp16": "P", "cluster": "X", "cluster-fp16": "X", "global-sync": "h", "global-sync-fp16": "h",
    "csr-row": "<", "csr-block": ">", "csr-cluster": "*", "auto": "o", "public-matched": "x", "public-library": "+",
}


def apply():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300,
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def order(labels):
    return sorted(labels, key=lambda l: PATH_ORDER.index(l) if l in PATH_ORDER else 99)
