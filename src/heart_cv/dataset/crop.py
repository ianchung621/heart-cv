from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class Cropper:
    """
    patient-volume cropper using 3D bbox
    """
    def __init__(
        self,
        bbox3d_df: pd.DataFrame | None = None,
        crop_dir: Path = Path("dataset/cropped"),
    ):
        self.crop_dir = Path(crop_dir)
        self.meta_path = self.crop_dir / "meta.json"

        if bbox3d_df is not None:
            required = {"pid", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
            if not required.issubset(bbox3d_df.columns):
                raise ValueError(f"bbox3d_df must include {required}")
            self.bbox_df = bbox3d_df.astype(int).copy()
        elif self.meta_path.exists():
            self.load_meta(self.meta_path)
        else:
            raise ValueError("Either bbox3d_df or existing meta.json must be provided.")

    # ------------------------------------------------------------ #
    # coordinate conversions
    # ------------------------------------------------------------ #
    @staticmethod
    def yolo_to_pixel(yolo: np.ndarray, W: int, H: int) -> np.ndarray:
        xc, yc, w, h = yolo.T
        x1 = (xc - w / 2) * W
        y1 = (yc - h / 2) * H
        x2 = (xc + w / 2) * W
        y2 = (yc + h / 2) * H
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def pixel_to_yolo(box: np.ndarray, W: int, H: int) -> np.ndarray:
        x1, y1, x2, y2 = box.T
        xc = (x1 + x2) / 2 / W
        yc = (y1 + y2) / 2 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        return np.stack([xc, yc, w, h], axis=-1)

    # ------------------------------------------------------------ #
    # coordinate transform using bbox_df
    # ------------------------------------------------------------ #
    def to_crop_label(self, yolo_label: np.ndarray, pid: int, img_shape: tuple[int, int]) -> np.ndarray:
        """Transform YOLO label (orig coords) → cropped coords using bbox_df."""
        W, H = img_shape
        row = self.bbox_df.loc[self.bbox_df.pid == pid].iloc[0]
        x1, y1, x2, y2 = row[["x_min", "y_min", "x_max", "y_max"]].astype(int)

        box = self.yolo_to_pixel(yolo_label, W, H)
        box[:, [0, 2]] -= x1
        box[:, [1, 3]] -= y1
        return self.pixel_to_yolo(box, x2 - x1, y2 - y1)

    def to_orig_label(self, yolo_crop: np.ndarray, pid: int, img_shape: tuple[int, int]) -> np.ndarray:
        """Transform YOLO label (cropped coords) → original coords using bbox_df."""
        W, H = img_shape
        row = self.bbox_df.loc[self.bbox_df.pid == pid].iloc[0]
        x1, y1, x2, y2 = row[["x_min", "y_min", "x_max", "y_max"]].astype(int)

        box = self.yolo_to_pixel(yolo_crop, x2 - x1, y2 - y1)
        box[:, [0, 2]] += x1
        box[:, [1, 3]] += y1
        return self.pixel_to_yolo(box, W, H)

    # ------------------------------------------------------------ #
    # single-slice cropper
    # ------------------------------------------------------------ #
    def _process_slice(
        self,
        pid: int,
        z: int,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        image_root: Path | None,
    ):
        pid_str = f"patient{pid:04d}"
        img_name = f"{pid_str}_{z:04d}.png"
        split = "training_image" if pid <= 50 else "testing_image"

        if image_root is None:
            image_root = Path("dataset/training_image") if pid <= 50 else Path("dataset/testing_image")

        src_img = image_root / pid_str / img_name
        dst_img = self.crop_dir / split / pid_str / img_name
        dst_img.parent.mkdir(parents=True, exist_ok=True)

        if not src_img.exists():
            return
        img = cv2.imread(str(src_img))
        if img is None:
            return

        crop = img[y1:y2, x1:x2]
        cv2.imwrite(str(dst_img), crop)

        # transform label only for training
        if pid <= 50:
            lbl_path = Path("dataset/training_label") / pid_str / img_name.replace(".png", ".txt")
            if lbl_path.exists():
                arr = np.loadtxt(lbl_path, ndmin=2)
                if arr.size > 0:
                    cls = arr[:, [0]]
                    yolo = arr[:, 1:5]
                    yolo_crop = self.to_crop_label(yolo, pid, (img.shape[1], img.shape[0]))
                    out = np.hstack([cls, yolo_crop])
                    dst_lbl = self.crop_dir / "training_label" / pid_str / lbl_path.name
                    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(dst_lbl, out, fmt="%.6f")

    # ------------------------------------------------------------ #
    # main cropper
    # ------------------------------------------------------------ #
    def crop_by_bbox3d(
        self,
        image_root: Path | None = None,
        max_workers: int = 8,
        save_meta: bool = True,
    ):
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for _, r in self.bbox_df.iterrows():
                pid = int(r.pid)
                x1, x2 = r.x_min, r.x_max
                y1, y2 = r.y_min, r.y_max
                for z in range(r.z_min, r.z_max + 1):
                    tasks.append(ex.submit(self._process_slice, pid, z, x1, x2, y1, y2, image_root))
            for t in tqdm(tasks, desc="Cropping"):
                t.result()

        if save_meta:
            self.save_meta(self.meta_path)

    # ------------------------------------------------------------ #
    # metadata persistence
    # ------------------------------------------------------------ #
    def save_meta(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = self.bbox_df.astype(int).to_dict(orient="records")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_meta(self, path: Path):
        with open(path, "r") as f:
            data = json.load(f)
        self.bbox_df = pd.DataFrame(data).astype(int)

    # ------------------------------------------------------------ #
    # inverse: load cropped predictions -> original pixels
    # ------------------------------------------------------------ #
        # ------------------------------------------------------------ #
    # inverse: load cropped predictions -> original pixels
    # ------------------------------------------------------------ #
    def load_cropped_label_dir(
        self,
        label_dir: Path,
        img_shape=(1024, 1024),
    ) -> pd.DataFrame:
        """
        Reproject YOLO txts from cropped labels back to original pixel coords.

        Returns
        -------
        pd.DataFrame
            Columns: ['img', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2']
        """
        label_dir = Path(label_dir)
        
        rows = []
        for txt_path in sorted(label_dir.rglob("*.txt")):
            name = txt_path.stem  # patient0001_0123
            pid = int(name.split("_")[0].replace("patient", ""))
            z = int(name.split("_")[1])
            if pid not in self.bbox_df.pid.values:
                continue

            arr = np.loadtxt(txt_path, ndmin=2)
            if arr.size == 0:
                continue

            cls = arr[:, 0]
            xc, yc, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            if arr.shape[1] > 5:
                conf = arr[:, 5]
            else:
                conf = np.full(len(arr),np.nan)
            yolo_crop = np.stack([xc, yc, w, h], axis=-1)
            yolo_orig = self.to_orig_label(yolo_crop, pid, img_shape)
            box = self.yolo_to_pixel(yolo_orig, *img_shape)

            for i in range(len(cls)):
                rows.append(
                    dict(
                        img=name,
                        cls=int(cls[i]),
                        conf=float(conf[i]),
                        x1=box[i, 0],
                        y1=box[i, 1],
                        x2=box[i, 2],
                        y2=box[i, 3],
                    )
                )

        df = pd.DataFrame(rows, columns=["img", "cls", "conf", "x1", "y1", "x2", "y2"])
        return df