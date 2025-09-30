import argparse
import time
import cv2
import os
import numpy as np
from ultralytics import YOLO


class MobilePhoneDetection:
    def __init__(self,
                 person_model_weights="yolov8s.pt",
                 phone_model_weights="yolov8s.pt",
                 conf_thresh=0.25,
                 iou_thresh=0.45,
                 device="cuda"):
        self.person_detector = YOLO(person_model_weights)
        self.phone_detector = YOLO(phone_model_weights)
        self.conf = conf_thresh
        self.iou_th = iou_thresh
        self.device = device

    # -------------------------- public predict API ---------------------------
    def predict(self, frame: np.ndarray, *, draw: bool = True, return_crops: bool = True):
        """Detect persons and phones and optionally draw annotated frame."""

        # 1. Person detection
        res_p = self.person_detector.predict(
            frame, classes=[0], conf=self.conf, iou=self.iou_th,
            device=self.device, verbose=False
        )[0]

        persons_det, crops, offs, crop_indices = [], [], [], []
        for b in res_p.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            s = float(b.conf[0])
            det_idx = len(persons_det)
            persons_det.append([x1, y1, x2, y2, s])
            if return_crops:
                width = max(x2 - x1, 1)
                expand = int(round(width * 0.20))
                frame_h, frame_w = frame.shape[:2]

                x1c = max(x1 - expand, 0)
                x2c = min(x2 + expand, frame_w)
                y1c = max(y1, 0)
                y2c = min(y2, frame_h)

                if x2c <= x1c or y2c <= y1c:
                    continue

                crops.append(frame[y1c:y2c, x1c:x2c].copy())
                offs.append((x1c, y1c))
                crop_indices.append(det_idx)

        # 2. Trackless association placeholders
        person_has_phone = [False] * len(persons_det)

        # 3. Phone detection inside crops
        phones = []
        if crops:
            res_ph = self.phone_detector.predict(
                crops, classes=[0], conf=self.conf, iou=self.iou_th, # set classes=[67] for model trained on COCO
                device=self.device, verbose=False
            )
            for r, (ox, oy), det_idx in zip(res_ph, offs, crop_indices):
                tid = det_idx
                for b in r.boxes:
                    px1, py1, px2, py2 = b.xyxy[0].cpu().numpy().astype(int)
                    s = float(b.conf[0])
                    person_has_phone[det_idx] = True
                    phones.append((px1 + ox, py1 + oy, px2 + ox, py2 + oy, s, tid))

        # 4. Drawing annotations
        annotated = None
        if draw:
            canvas = frame.copy()
            for idx, has_phone in enumerate(person_has_phone):
                if not has_phone:
                    continue
                x1, y1, x2, y2, _ = persons_det[idx]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(canvas, f"ID {idx} | Phone", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            annotated = canvas

        tracked_persons = []
        for idx, has_phone in enumerate(person_has_phone):
            if not has_phone:
                continue
            x1, y1, x2, y2, _ = persons_det[idx]
            tracked_persons.append((x1, y1, x2, y2, idx))
        return tracked_persons, phones, (crops if return_crops else None), annotated

    # ------------------------- video util -----------------------------------
    def run_video(self, source: str | int = 0, *, show_fps: bool = True,
                  save_path: str | None = None, show_streaming: bool = True) -> None:
        TITLE = "TonAI Computer Vision"
        total_frames = 0
        if 'rtsp://' in str(source):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;60000000"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 0)
        else:
            cap = cv2.VideoCapture(source)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        if show_streaming:
            cv2.namedWindow(TITLE, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))

        try:
            print("Object detection is running...")
            while True:
                if total_frames > 0:
                    print(f"Progress: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames} frames", end='\r')
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                *_, annotated = self.predict(frame, draw=True)
                dt = time.time() - t0

                if show_fps and annotated is not None:
                    fps = 1.0 / (dt + 1e-6)
                    cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if writer and annotated is not None:
                    writer.write(annotated)

                if show_streaming and annotated is not None:
                    cv2.imshow(TITLE, annotated)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            print("Finished!")
            cap.release()
            if writer:
                writer.release()
            if show_streaming:
                cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=0)
    parser.add_argument("--person_model_weights", type=str, default="weights/yolo12n.pt")
    parser.add_argument("--phone_model_weights", type=str, default="weights/phone_yolov9m_26092025.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.35)
    parser.add_argument("--iou_thresh", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show_fps", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--show_streaming", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipe = MobilePhoneDetection(person_model_weights=args.person_model_weights,
                                phone_model_weights=args.phone_model_weights,
                                conf_thresh=args.conf_thresh,
                                iou_thresh=args.iou_thresh,
                                device=args.device)
    pipe.run_video(source=args.source, # Source can be RTSP url or file path
                   show_streaming=args.show_streaming,
                   save_path=args.save_path)
