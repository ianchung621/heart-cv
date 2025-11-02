import networkx as nx
import pandas as pd
import numpy as np

from typing import Callable
from .utils import add_pid_z_paths

def build_patient_graph(
    df_pred: pd.DataFrame,
    pid: int,
    edge_mode: str = "iou",
    min_weight: float = 0.05,
    best_weight: bool = False,
):
    """
    slice-by-slice box connectivity for a single patient.

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
    min_weight : float, default=0.05
        Minimum edge weight to draw; suppresses weak connections.

    Returns
    -------
    G : networkx.DiGraph
        Directed layered graph with node attributes and weighted edges.
    """

    # --- preprocess --- #
    df_pred = add_pid_z_paths(df_pred.copy())
    df_pid = df_pred[df_pred.pid == pid]
    df_pid = df_pid.sort_values("z").reset_index(drop=True)
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
        G.add_node(i, z=r.z, conf=r.conf, row=r.to_dict())

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
    return G

def parse_tubes_from_graph(G: nx.DiGraph, pid: int) -> pd.DataFrame:
    """
    Parse a patient graph into tube-level DataFrame.
    Each weakly connected component is treated as one tube.
    """
    tubes = []
    for tube_id, comp in enumerate(nx.weakly_connected_components(G)):
        for node in comp:
            row = G.nodes[node]["row"].copy()
            row["pid"] = pid
            row["tube_id"] = tube_id
            tubes.append(row)
    if not tubes:
        return pd.DataFrame()
    return pd.DataFrame(tubes).sort_values(["tube_id", "z"]).reset_index(drop=True)


def build_patient_graph_and_tube_dict(
    df_pred: pd.DataFrame,
    edge_mode: str = "iou",
    min_weight: float = 0.5,
    best_weight: bool = False,
) -> dict[int, tuple[nx.DiGraph, pd.DataFrame]]:
    """
    Build patient-wise graphs and their tube DataFrames.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Must contain ['pid','z','x1','y1','x2','y2','conf'].
    edge_mode : {'iou','center'}, default='iou'
        Edge weight definition.
    min_weight : float, default=0.05
        Minimum edge weight to include.
    best_weight : bool, default=False
        If True, connect each node only to its best match in the next slice.

    Returns
    -------
    dict[int, (nx.DiGraph, pd.DataFrame)]
        Mapping: patient id → (graph, tube_df)
    """
    pid_dict: dict[int, tuple[nx.DiGraph, pd.DataFrame]] = {}
    for pid in sorted(df_pred["pid"].unique()):
        G = build_patient_graph(
            df_pred=df_pred,
            pid=pid,
            edge_mode=edge_mode,
            min_weight=min_weight,
            best_weight=best_weight,
        )
        df_tube = parse_tubes_from_graph(G, pid)
        pid_dict[pid] = (G, df_tube)
    return pid_dict

def aggregate_tube_data(
    tube_data: dict[int, tuple[nx.DiGraph, pd.DataFrame]],
    func: Callable[[pd.DataFrame, int], pd.DataFrame],
) -> pd.DataFrame:
    """
    Apply a user-defined aggregation function to all patient tube DataFrames
    and concatenate the results.

    Parameters
    ----------
    tube_data : dict[int, tuple[nx.DiGraph, pd.DataFrame]]
        Mapping {pid: (G, df_tube)} where df_tube includes columns:
        ['img','cls','conf','x1','y1','x2','y2','pid','z','tube_id']
    func : callable
        Function applied to each patient's df_tube.
        Must accept arguments (df_tube, pid) and return a DataFrame.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all patient-level results.
    """
    results = []
    for pid, (_, df_tube) in tube_data.items():
        if df_tube is None or df_tube.empty:
            continue
        df_result = func(df_tube, pid)
        if df_result is not None and not df_result.empty:
            results.append(df_result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)