import networkx as nx
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..postprocessing import add_pid_z_paths
from ..metric import compute_iou



def plot_patient_tube_graph(
    df_pred: pd.DataFrame,
    pid: int,
    df_gt: pd.DataFrame | None = None,
    edge_mode: str = "iou",
    min_weight: float = 0.5,
    figsize=(6, 25),
    savefn: str = "tube.pdf",
    best_weight: bool = False,
):
    """
    Visualize slice-by-slice box connectivity for a single patient as a *vertical layered graph*.

    Each row (y-axis) corresponds to one z-slice; nodes in that row are the predicted boxes
    for that slice (sorted by area, left→right). Edges connect boxes between adjacent z-slices
    with weights given by IoU or spatial proximity.

    Node color indicates detection correctness (IoU > 0.5 → lime, else red).
    Node text shows the raw confidence value.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Prediction DataFrame containing columns:
        ['pid','z','x1','y1','x2','y2','conf'].
    df_gt : pd.DataFrame
        Ground-truth boxes for the same patient (used by compute_iou).
    pid : int
        Patient ID to visualize.
    edge_mode : {'iou','center'}, default='iou'
        Edge weight definition:
        - 'iou':  IoU overlap between boxes (higher → stronger connection)
        - 'center': exp(-distance/20)
    min_weight : float, default=0.5
        Minimum edge weight to draw; suppresses weak connections.
    figsize : tuple, default=(8,12)
        Figure size for the vertical layout.
    savefn : str, default='tube.pdf'
        Output path for saving the figure (if None, display interactively).

    Returns
    -------
    G : networkx.DiGraph
        Directed layered graph with node attributes and weighted edges.
    """

    # --- preprocess --- #
    df_pred = add_pid_z_paths(df_pred.copy())
    df_pid = df_pred[df_pred.pid == pid]
    df_pid = df_pid.sort_values("z").reset_index(drop=True)
    if not df_gt is None:
        df_gt = add_pid_z_paths(df_gt.copy())
        df_gt = df_gt[df_gt.pid == pid]
        df_pid = compute_iou(df_pid, df_gt)  # assume adds 'iou' column
    df_pid["area"] = (df_pid.x2 - df_pid.x1) * (df_pid.y2 - df_pid.y1)
    zs = sorted(df_pid["z"].unique())
    G = nx.DiGraph()

    # --- helper functions --- #
    def box_iou(a, b):
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        union = (a.x2 - a.x1)*(a.y2 - a.y1) + (b.x2 - b.x1)*(b.y2 - b.y1) - inter
        return inter / union if union > 0 else 0.0

    def box_center(b):
        return np.array([(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2])

    def center_dist(a, b):
        return np.linalg.norm(box_center(a) - box_center(b))

    # --- nodes --- #
    for i, r in df_pid.iterrows():
        if not df_gt is None:
            G.add_node(i, z=r.z, conf=r.conf, area=r.area, iou=r.iou)
        else:
            G.add_node(i, z=r.z, conf=r.conf, area=r.area)

    # --- edges --- #
    for z_prev, z_next in zip(zs[:-1], zs[1:]):
        df_prev = df_pid[df_pid.z == z_prev]
        df_next = df_pid[df_pid.z == z_next]

        for i, a in df_prev.iterrows():
            # compute weights to all next-slice boxes
            weights = []
            for j, b in df_next.iterrows():
                if edge_mode == "iou":
                    w = box_iou(a, b)
                else:
                    d = center_dist(a, b)
                    w = np.exp(-d / 20)
                weights.append((j, w))

            # optionally keep only best-weight edge
            if best_weight:
                j_best, w_best = max(weights, key=lambda x: x[1])
                if w_best >= min_weight:
                    G.add_edge(i, j_best, weight=w_best)
            else:
                for j, w in weights:
                    if w >= min_weight:
                        G.add_edge(i, j, weight=w)

    # --- vertical layered layout (sorted by area descending) --- #
    pos = {}
    node_spacing = 2.5
    layer_spacing = 3

    for z in zs:
        df_layer = df_pid[df_pid.z == z].sort_values("area", ascending=False)
        for k, (idx, r) in enumerate(df_layer.iterrows()):
            # horizontally spread boxes in same z
            pos[idx] = (k * node_spacing, -z * layer_spacing)

    # --- plot --- #
    plt.figure(figsize=figsize)
    if not df_gt is None:
        node_colors = ["lime" if d["iou"] > 0.5 else "red" for _, d in G.nodes(data=True)]
    else:
        node_colors = "cyan"
    node_sizes = [d["area"] * 0.3 for _, d in G.nodes(data=True)]

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="grey",
        alpha=0.85,
        arrows=True,
        connectionstyle="arc3,rad=0.0",
    )

    # --- add edge weights (IoU) --- #
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True) if 'weight' in d}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="grey",
        label_pos=0.5,  # position along edge (0→source, 1→target)
        rotate=False,
        bbox=dict(facecolor="none", edgecolor="none", alpha=0)
    )

    # --- add confidence text --- #
    for n, (x, y) in pos.items():
        conf = G.nodes[n]["conf"]
        plt.text(
            x,
            y + 0.2,
            f"{conf:.2f}",
            fontsize=8,
            fontweight='bold',
            ha="center",
            va="bottom",
            color="black",
            alpha=0.9,
        )

    # --- add z labels & horizontal separators --- #
    x_min = min(x for x, _ in pos.values())
    z_max, z_min = df_gt.z.max(), df_gt.z.min()
    plt.text(x_min - 0.8, -(min(zs)-1)*layer_spacing,"z", fontsize=9, ha="right", va="center", color="blue",)
    for z in zs:
        z_boundary = (z == z_min) or (z == z_max)
        y = -z * layer_spacing
        plt.axhline(y=y - layer_spacing / 2, color="gray", linestyle="--", lw=0.8, alpha=0.5)
        plt.text(
            x_min - 0.8,
            y,
            f"{z}",
            fontsize=9,
            fontweight = "bold" if z_boundary else "normal",
            ha="right",
            va="center",
            color="blue",
        )

    plt.title(f"Box layered graph ({edge_mode} > {min_weight})")
    plt.xlabel("Boxes (sorted by area, left→right)")
    plt.ylabel("z-slice (top→bottom)")
    plt.margins(0.05, tight=True)

    # --- fix axis scaling for single-column (narrow) graphs --- #
    xs, ys = zip(*pos.values())
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    if x_range < 1e-6:  # single column of nodes
        pad = node_spacing * 0.6
        plt.xlim(-pad, pad)
    else:
        plt.xlim(min(xs) - node_spacing, max(xs) + node_spacing)

    plt.ylim(min(ys) - layer_spacing, max(ys) + layer_spacing)
    plt.gca().set_aspect("equal", adjustable="box")

    if savefn:
        plt.savefig(savefn, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return G

def plot_all_patient_tube_graph(
    df_pred: pd.DataFrame,
    df_gt: pd.DataFrame | None = None,
    edge_mode: str = "iou",
    min_weight: float = 0.5,
    figsize=(6, 25),
    save_dir: Path = Path("tube"),
    best_weight: bool = False,
):
    df_pred = add_pid_z_paths(df_pred.copy())
    pids = df_pred.pid.unique()
    save_dir.mkdir(parents=True, exist_ok=True)
    for pid in tqdm(pids, desc=f"Ploting graph to {save_dir}"):
        plot_patient_tube_graph(
            df_pred,
            pid,
            df_gt,
            edge_mode,
            min_weight,
            figsize,
            save_dir / f"{pid:04d}.pdf",
            best_weight,
        )
