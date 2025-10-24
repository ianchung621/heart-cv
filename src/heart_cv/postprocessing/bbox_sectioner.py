import pandas as pd
import numpy as np
from typing import Callable
from .utils import add_pid_z_paths

class BBoxSectioner:
    """
    Divide predictions along the z-axis (slice dimension) into anatomical sections
    like 'head', 'body', and 'tail', based on each patient's 3D bounding box.

    This allows applying different postprocessing functions (e.g., NMS, confidence boosting)
    to each section independently.

    Parameters
    ----------
    ssection_bounds : dict[str, tuple[float, float]] | tuple[float, float], optional
        Fractional z-ranges for each section relative to [z_min, z_max].
        Accepts:
            - dict: custom boundaries for each section.
            - tuple: (head_end, body_end), used to auto-derive head/body/tail.
        Default:
            {
                "head": (0.0, 0.3),
                "body": (0.3, 0.7),
                "tail": (0.7, 1.0)
            }
    """

    def __init__(self, section_bounds: dict[str, tuple[float, float]] | tuple[float, float] | None = None):
        if section_bounds is None:
            self.section_bounds = {
                "head": (0.0, 0.3),
                "body": (0.3, 0.7),
                "tail": (0.7, 1.0),
            }
        elif isinstance(section_bounds, tuple) and len(section_bounds) == 2:
            h_end, b_end = section_bounds
            if not (0.0 <= h_end < b_end <= 1.0):
                raise ValueError("section_bounds tuple must satisfy 0.0 <= head < body <= 1.0")
            self.section_bounds = {
                "head": (0.0, h_end),
                "body": (h_end, b_end),
                "tail": (b_end, 1.0),
            }
        elif isinstance(section_bounds, dict):
            self.section_bounds = section_bounds
        else:
            raise TypeError("section_bounds must be dict, tuple, or None")

    def assign_section(self, df_pred: pd.DataFrame, bbox_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each prediction a 'section' label based on its z-position
        relative to the 3D bounding box.

        Parameters
        ----------
        df_pred : pd.DataFrame
            Predictions with columns ['pid', 'z'].
        bbox_df : pd.DataFrame
            Bounding boxes with columns ['pid', 'z_min', 'z_max'].

        Returns
        -------
        pd.DataFrame
            Copy of df_pred with an added 'section' column.
        """
        if "pid" not in df_pred.columns:
            df_pred = add_pid_z_paths(df_pred)
        merged = df_pred.merge(bbox_df[["pid", "z_min", "z_max"]], on="pid", how="left")
        merged["z_portion"] = (merged["z"] - merged["z_min"]) / (merged["z_max"] - merged["z_min"])
        merged["z_portion"] = merged["z_portion"].clip(0, 1)

        def get_section(p):
            for name, (lo, hi) in self.section_bounds.items():
                if lo <= p < hi:
                    return name
            return "unknown"

        merged["section"] = merged["z_portion"].apply(get_section)
        return merged

    def apply_by_section(
        self,
        df_pred: pd.DataFrame,
        bbox_df: pd.DataFrame,
        func_map: dict[str, Callable[[pd.DataFrame], pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        Apply different postprocessing functions to each section.

        Parameters
        ----------
        df_pred : pd.DataFrame
            Predictions with ['pid', 'z', ...].
        bbox_df : pd.DataFrame
            Bounding boxes with ['pid', 'z_min', 'z_max'].
        func_map : dict[str, Callable]
            Mapping from section name to postprocessing function.
            Example:
                {
                    "head": func_head,
                    "body": func_body,
                    "tail": func_tail
                }

        Returns
        -------
        pd.DataFrame
            Concatenated postprocessed results from all sections.
        """

        # --- drop any pre-existing bbox columns ---
        bbox_cols = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        df_pred = df_pred.drop(columns=[c for c in bbox_cols if c in df_pred.columns])

        df_labeled = self.assign_section(df_pred, bbox_df)
        results = []

        for section, (lo, hi) in self.section_bounds.items():
            subset = df_labeled[df_labeled["section"] == section]
            if len(subset) == 0:
                continue

            func = func_map.get(section, lambda df: df)
            results.append(func(subset))

        # merge and keep consistent order
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return df_labeled