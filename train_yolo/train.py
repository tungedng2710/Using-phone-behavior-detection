#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import argparse

import mlflow                       # noqa: F401  (just to ensure import works)
from minio import Minio
from ultralytics import YOLO, settings

# MinIO defaults
UPLOAD_RESULTS_TO_MINIO = True
MINIO_ENDPOINT = '0.0.0.0:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET = 'iva'
MINIO_PREFIX = 'yolo_runs'

# ---------------- Ultralytics + MLflow ----------------
settings.update({"mlflow": True})

# ---------------- ensure datasets dir exists ----------------
os.makedirs("./datasets", exist_ok=True)


def upload_folder_to_minio(client: Minio, bucket: str,
                           local_dir: Path, prefix: str = "") -> None:
    """Recursively upload `local_dir` to `<bucket>/<prefix>` in MinIO."""
    for root, _, files in os.walk(local_dir):
        for fname in files:
            fpath = Path(root, fname)
            object_name = str(Path(prefix, fpath.relative_to(local_dir))).replace("\\", "/")
            client.fput_object(bucket, object_name, str(fpath))


class YOLOTrainer:
    def __init__(self, model_path: str, data_path: str):
        assert model_path, "model_path must not be empty"
        assert data_path,  "data_path must not be empty"
        self.model_path = model_path

        # ── MinIO connection (optional) ──
        self.client = None
        if UPLOAD_RESULTS_TO_MINIO:
            try:
                client = Minio(MINIO_ENDPOINT,
                               access_key=MINIO_ACCESS_KEY,
                               secret_key=MINIO_SECRET_KEY,
                               secure=False)
                client.list_buckets()                # sanity check
                if not client.bucket_exists(MINIO_BUCKET):
                    client.make_bucket(MINIO_BUCKET)
                self.client = client
                print(f"[MinIO] Connected to {MINIO_ENDPOINT}")
            except Exception as exc:
                print(f"[MinIO] Disabled → {exc}")

        self.model = YOLO(model_path)
        self.data  = data_path

    # -------------------------------------------------
    def train(self, **kwargs):
        """
        Any keyword accepted by ultralytics.YOLO.train may be forwarded via **kwargs
        (epochs, imgsz, batch, device, etc.).
        """
        # auto_set_name handled here to keep __main__ simple
        if kwargs.pop("auto_set_name", False):
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_tag = Path(self.model_path).stem
            data_tag  = Path(self.data).parent.name or Path(self.data).stem
            kwargs["name"] = f"{model_tag}_{data_tag}_{kwargs.get('epochs', '??')}eps_" \
                             f"{kwargs.get('imgsz', '??')}_{ts}"

        results = self.model.train(data=self.data, **kwargs)

        # upload artifacts
        if self.client:
            run_dir = Path(results.save_dir)
            prefix  = f"{MINIO_PREFIX}/{run_dir.name}"
            upload_folder_to_minio(self.client, MINIO_BUCKET, run_dir, prefix)
            print(f"Uploaded artifacts → s3://{MINIO_BUCKET}/{prefix}/")

        return results


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a YOLO model with optional MLflow + MinIO logging")

    # Required
    p.add_argument("--model-path", required=True,
                   help="Path to the YOLO model weights (e.g. ./weights/yolov8l.pt)")
    p.add_argument("--data-path",  required=True,
                   help="Ultralytics YAML dataset file (e.g. ./datasets/data.yaml)")

    # Common training knobs
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz",  type=int, default=640, help="Image size")
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--device", default="0",
                   help="GPU device id(s) – 'cpu', '0', or '0,1'")

    # Quality-of-life toggles
    p.add_argument("--pretrained",  action="store_true", help="Use pretrained weights")
    p.add_argument("--resume",      action="store_true", help="Resume previous run")
    p.add_argument("--cache",       action="store_true", help="Cache images")
    p.add_argument("--cos-lr",      action="store_true", help="Use cosine LR schedule")
    p.add_argument("--auto-set-name", action="store_true",
                   help="Auto-generate run name with timestamp")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert device string -> proper format (int or list[int]/str)
    device_arg = args.device
    if "," in device_arg:
        device_arg = [int(d) for d in device_arg.split(",") if d.strip().isdigit()]
    elif device_arg.isdigit():
        device_arg = int(device_arg)           # single GPU id
    # else leave as 'cpu'

    trainer = YOLOTrainer(model_path=args.model_path,
                          data_path=args.data_path)

    trainer.train(epochs=args.epochs,
                  imgsz=args.imgsz,
                  batch=args.batch,
                  device=device_arg,
                  pretrained=args.pretrained,
                  resume=args.resume,
                  cache=args.cache,
                  cos_lr=args.cos_lr,
                  auto_set_name=args.auto_set_name)