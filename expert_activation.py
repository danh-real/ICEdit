import ast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Data loading ──────────────────────────────────────────────────────────────
df = pd.read_csv("ablation_AnyEdit.csv", index_col=0)

weights = pd.DataFrame(
    df["route_weight"].apply(ast.literal_eval).tolist(),
    index=df.index,
    columns=["Expert0", "Expert1", "Expert2", "Expert3"],
)
_base_cols = ["image_path", "edit_type", "layer", "timestep"] + (["route_path"] if "route_path" in df.columns else [])
df_expand = pd.concat([df[_base_cols], weights], axis=1)

# ── Layer ordering (Flux architecture) ───────────────────────────────────────
N_TB  = 19   # transformer_blocks.0-18
N_STB = 38   # single_transformer_blocks.0-37
TB_LAYERS  = [f"transformer_blocks.{i}"        for i in range(N_TB)]
STB_LAYERS = [f"single_transformer_blocks.{i}" for i in range(N_STB)]
ALL_LAYERS = TB_LAYERS + STB_LAYERS
LAYER_IDX  = {l: i for i, l in enumerate(ALL_LAYERS)}

EXPERT_COLS = ["Expert0", "Expert1", "Expert2", "Expert3"]
N_LAYERS    = len(ALL_LAYERS)     # 57
N_TIMESTEPS = df["timestep"].nunique()  # 28

EXPERT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # 0-3
EXPERT_CMAP   = plt.cm.colors.ListedColormap(EXPERT_COLORS)


# # ── Parse route_path → 16 flat columns ───────────────────────────────────────
_PATH_COLS = [f"p{i}{j}" for i in range(4) for j in range(4)]
# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Flow diagram: token routing paths across layers
# (parallel-coordinates style, similar to "Independent Routing" diagram)
# One subplot per edit type, filtered to "resize" and "style_change".
# Line thickness ∝ normalised fraction of tokens following that connection.
# ─────────────────────────────────────────────────────────────────────────────

_FLOW_TYPES   = ["resize", "style_change", "remove", "add", "color_alter"]
_CASE_ID = [
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000478670.jpg",
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000003623.jpg",
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000330979.jpg",
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000066652.jpg",
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000552093.jpg",
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000555627.jpg",
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000064186.jpg",
]
_SUCCESS_CASE = [    
    "/data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000478670.jpg",
]
_FLOW_COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # one per expert
_N_EXP        = 4
_EXP_X        = np.array([0.0, 1.0, 2.0, 3.0])   # expert column x-positions
_STEP         = 1.0    # vertical distance between consecutive layers
_BAR_H        = 0.28   # height of each layer bar
_BOX_W        = 0.42   # width of each expert box inside the bar
_BOX_H        = 0.20   # height of each expert box


def _bezier(p0, p1, p2, p3, n=80):
    t  = np.linspace(0, 1, n)
    bx = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
    by = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
    return bx, by


def _draw_flow_ax(ax, route_data, layers, n_tb, title):
    """Draw a single flow-diagram subplot.

    route_data : dict  layer_name -> (4,4) normalised numpy array
    layers     : ordered list of all layer names
    """
    n_layers = len(layers)

    def ly(i):
        return -(i * _STEP)

    # ── Draw layer bars + expert boxes ──────────────────────────────────────
    for i, lname in enumerate(layers):
        yc = ly(i)
        ax.add_patch(mpatches.FancyBboxPatch(
            (_EXP_X[0] - _BOX_W/2 - 0.06, yc - _BAR_H/2),
            _EXP_X[-1] - _EXP_X[0] + _BOX_W + 0.12, _BAR_H,
            boxstyle="round,pad=0.03", linewidth=0.4,
            facecolor="#F0F0F0", edgecolor="#BBBBBB", zorder=2,
        ))
        for e in range(_N_EXP):
            ax.add_patch(mpatches.FancyBboxPatch(
                (_EXP_X[e] - _BOX_W/2, yc - _BOX_H/2),
                _BOX_W, _BOX_H,
                boxstyle="round,pad=0.02", linewidth=0.5,
                facecolor="white", edgecolor="#999999", zorder=3,
            ))
        label = f"TB.{i}" if i < n_tb else f"STB.{i - n_tb}"
        ax.text(_EXP_X[0] - _BOX_W/2 - 0.12, yc, label,
                ha="right", va="center", fontsize=4.5, color="#555555")

    # ── Draw connections ─────────────────────────────────────────────────────
    for i in range(1, n_layers):
        lname = layers[i]
        if lname not in route_data:
            continue
        mat = route_data[lname]

        yf   = ly(i - 1) - _BAR_H / 2
        yt   = ly(i)     + _BAR_H / 2
        ymid = (yf + yt) / 2

        for fe in range(_N_EXP):
            for te in range(_N_EXP):
                w = float(mat[fe, te])
                if w < 4e-3:
                    continue
                xf, xt = _EXP_X[fe], _EXP_X[te]
                lw     = max(0.4, w * 14)
                color  = _FLOW_COLORS[fe]

                bx, by = _bezier(
                    [xf, yf], [xf, ymid], [xt, ymid], [xt, yt],
                )
                ax.plot(bx, by, color=color, linewidth=lw,
                        alpha=0.55, solid_capstyle="round", zorder=1)

                ds = max(1.5, w * 18)
                ax.plot(xf, yf, "o", color=color, ms=ds, zorder=4, alpha=0.85)
                ax.plot(xt, yt, "o", color=color, ms=ds, zorder=4, alpha=0.85)

    # expert labels at top and right side
    for e in range(_N_EXP):
        ax.text(_EXP_X[e], ly(0) + _BAR_H/2 + 0.15, f"E{e}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=_FLOW_COLORS[e])
        ax.text(_EXP_X[e], ly(n_layers - 1) - _BAR_H/2 - 0.15, f"E{e}",
                ha="center", va="top", fontsize=7, fontweight="bold",
                color=_FLOW_COLORS[e])

    for i, lname in enumerate(layers):
        label = f"TB.{i}" if i < n_tb else f"STB.{i - n_tb}"
        ax.text(_EXP_X[-1] + _BOX_W/2 + 0.12, ly(i), label,
                ha="left", va="center", fontsize=4.5, color="#555555")

    ax.set_title(title, fontsize=11, pad=10, fontweight="bold")
    ax.set_xlim(_EXP_X[0] - 1.1, _EXP_X[-1] + 1.1)
    ax.set_ylim(ly(n_layers - 1) - _STEP * 0.65, ly(0) + _STEP * 0.65)
    ax.axis("off")


# ── Aggregate route_path per (edit_type, layer) ───────────────────────────────
# _df_flow = df[df["edit_type"].isin(_FLOW_TYPES)].copy()
_df_flow = df[df["image_path"].isin(_CASE_ID)].copy()

_pf2  = _df_flow["route_path"].apply(lambda s: [v for row in ast.literal_eval(s) for v in row])
_pdf2 = pd.DataFrame(_pf2.tolist(), index=_df_flow.index, columns=_PATH_COLS)
_df_flow2 = pd.concat([_df_flow[["image_path", "edit_type", "layer"]], _pdf2], axis=1)
_df_flow2 = _df_flow2[_df_flow2["layer"] != "transformer_blocks.0"]
_df_flow2["success"] = _df_flow2["image_path"].isin(_SUCCESS_CASE)

# _agg_flow = _df_flow2.groupby(["edit_type", "layer"])[_PATH_COLS].sum()
_agg_flow = _df_flow2.groupby(["success", "layer"])[_PATH_COLS].sum()

def _get_route_data(etype):
    """Return dict: layer_name -> (4,4) fraction-normalised numpy array."""
    data = {}
    if etype not in _agg_flow.index.get_level_values("success"): #NOTE
        return data
    sub = _agg_flow.loc[etype]
    for lname, row in sub.iterrows():
        mat   = row.values.reshape(_N_EXP, _N_EXP).astype(float)
        total = mat.sum()
        if total > 0:
            mat /= total
        data[lname] = mat
    return data

AX_COLS = [True, False]
# ── Build figure ──────────────────────────────────────────────────────────────
_fig_h  = max(20, N_LAYERS * _STEP * 0.38)
fig6, axes6 = plt.subplots(
    1, len(AX_COLS),
    figsize=(len(AX_COLS) * 5.5, _fig_h),
    constrained_layout=True,
)

for ax_i, etype in enumerate(AX_COLS):
    _draw_flow_ax(axes6[ax_i], _get_route_data(etype), ALL_LAYERS, N_TB, title=f"success={etype}")

_leg = [mpatches.Patch(color=_FLOW_COLORS[e], label=f"from Expert {e}")
        for e in range(_N_EXP)]
fig6.legend(handles=_leg, loc="lower center", ncol=_N_EXP,
            fontsize=9, title="Line colour = source expert")
fig6.suptitle(
    "Token Routing Flow — Expert-to-Expert Paths Across Layers\n"
    "(line width ∝ fraction of tokens following that connection)",
    fontsize=12,
)
fig6.savefig("route_path_flow_resize.png", dpi=150, bbox_inches="tight")
print("Saved route_path_flow_resize.png")
plt.show()
