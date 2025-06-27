import os
import sys
from pathlib import Path

import mlflow
from minio import Minio
from ultralytics import YOLO, settings


# MinIO configuration
UPLOAD_RESULTS_TO_MINIO = True
MINIO_ENDPOINT = '0.0.0.0:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET = 'iva'
MINIO_PREFIX = 'yolo_runs'
FOLDER_PATH = './runs'
# Use MLFlow for monitoring
settings.update({"mlflow": True})

def upload_folder_to_minio(client: Minio, bucket: str, local_dir: Path, prefix: str = "") -> None:
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
        self.client = None # default: no object storage
        if UPLOAD_RESULTS_TO_MINIO:
            try:
                client = Minio(
                    MINIO_ENDPOINT,
                    access_key=MINIO_ACCESS_KEY,
                    secret_key=MINIO_SECRET_KEY,
                    secure=False 
                )
                client.list_buckets()
                
                if not client.bucket_exists(MINIO_BUCKET):
                    client.make_bucket(MINIO_BUCKET)

                self.client = client
                print(f"[MinIO] Connected to {MINIO_ENDPOINT}")

            except Exception as exc:
                print(f"[MinIO] Disabled -> {exc}")

        else:
            print("[MinIO] Package not installed or endpoint not set - skipping")

        self.model = YOLO(model_path)
        self.data  = data_path
        
    def train(self, epochs=100, imgsz=640, batch=32, device=[0], cache=False,
                workers=8, project=None, name=None, pretrained=True, resume=False,
                optimizer="auto", classes=None, lr0=0.01, lrf=0.01, cos_lr=False,
                momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, auto_set_name=False):
        if auto_set_name:
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_tag = os.path.basename(self.model_path).split('.')[0]
            data_tag  = os.path.basename(self.data).split('.')[0]
            name = f"{model_tag}_{data_tag}_{epochs}eps_{imgsz}_{ts}"
        results = self.model.train(data=self.data,
                                    batch=batch,
                                    epochs=epochs,
                                    imgsz=imgsz,
                                    device=device,
                                    cache=cache,
                                    workers=workers,
                                    project=project,
                                    name=name,
                                    pretrained=pretrained,
                                    resume=resume,
                                    optimizer=optimizer,
                                    classes=classes,
                                    lr0=lr0,
                                    lrf=lrf,
                                    cos_lr=cos_lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    warmup_epochs=warmup_epochs)
        run_dir = Path(results.save_dir)
        prefix  = f"{MINIO_PREFIX}/{run_dir.name}"
        upload_folder_to_minio(self.client, MINIO_BUCKET, run_dir, prefix)
        print(f"Uploaded artifacts to s3://{MINIO_BUCKET}/{prefix}/")
        return results


if __name__ == "__main__":
    if not os.path.exists("./datasets"):
        os.makedirs("./datasets")
    trainer = YOLOTrainer(model_path='./weights/yolo11l.pt',
                          data_path="./datasets/mobile_phone_v1.2/data.yaml")
    trainer.train(epochs=15, batch=32, device=1,
                  pretrained=True, auto_set_name=True)