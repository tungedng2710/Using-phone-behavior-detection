import argparse
import time
import cv2
import os
import numpy as np
from ultralytics import YOLO

from utils.utils import iou_matrix
from utils.sort_tracker import Sort


class MobilePhoneDetection:
    def __init__(self,
                 person_model_weights="yolov8s.pt",
                 phone_model_weights="yolov8s.pt",
                 conf_thresh=0.25,
                 iou_thresh=0.45,
                 device="cuda"):
        self.person_detector = YOLO(person_model_weights)
        self.phone_detector = YOLO(phone_model_weights)
        self.tracker = Sort()
        self.conf = conf_thresh
        self.iou_th = iou_thresh
        self.device = device
        self._mem: list[dict] = []
        self.retention_secs = 5

    # -------------------------- public predict API ---------------------------
    def predict(self, frame: np.ndarray, *, draw: bool = True, return_crops: bool = True):
        """Detect persons and phones and optionally draw annotated frame."""
        now = time.time()

        # 1. Person detection
        res_p = self.person_detector.predict(
            frame, classes=[0], conf=self.conf, iou=self.iou_th,
            device=self.device, verbose=False
        )[0]

        persons_det, crops, offs = [], [], []
        for b in res_p.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            s = float(b.conf[0])
            persons_det.append([x1, y1, x2, y2, s])
            if return_crops:
                # --- parameters you may want to tune -----------------------------
                ZOOM_RATIO      = 1.0   # keep 80 % of the shortest side
                TORSO_CENTER_Y  = 0.55   # 0 = top of bbox, 1 = bottom → 0.55 ≈ torso
                # -----------------------------------------------------------------

                w, h = x2 - x1, y2 - y1

                # horizontal centre stays the same
                cx = x1 + w * 0.5
                # but vertical centre is shifted a bit downward toward the torso
                cy = y1 + h * TORSO_CENTER_Y

                # side length of the (zoomed-in) square crop
                side = int(min(w, h) * ZOOM_RATIO)
                side = max(side, 1)  # avoid zero

                # tentative crop coordinates
                x1c = int(cx - side / 2)
                y1c = int(cy - side / 2)
                x2c = x1c + side
                y2c = y1c + side

                # keep the square fully inside the frame by translating it if needed
                if x1c < 0:
                    x2c -= x1c
                    x1c = 0
                if y1c < 0:
                    y2c -= y1c
                    y1c = 0
                if x2c > frame.shape[1]:
                    x1c -= (x2c - frame.shape[1])
                    x2c = frame.shape[1]
                if y2c > frame.shape[0]:
                    y1c -= (y2c - frame.shape[0])
                    y2c = frame.shape[0]

                # final clamp (in case the frame is smaller than the desired crop)
                x1c, y1c = max(x1c, 0), max(y1c, 0)
                x2c = min(x2c, frame.shape[1])
                y2c = min(y2c, frame.shape[0])

                crops.append(frame[y1c:y2c, x1c:x2c].copy())
                offs.append((x1c, y1c))

        # 2. Tracking persons using SORT
        det_array = np.array(persons_det) if persons_det else np.empty((0, 5))
        tracks = self.tracker.update(det_array)
        track_boxes = {int(t[4]): t[:4].astype(int) for t in tracks}

        # Assign detection index to track id via IoU
        id_by_det_idx = {}
        if persons_det:
            p_boxes = np.array([p[:4] for p in persons_det])
            for tid, box in track_boxes.items():
                ious = iou_matrix(np.asarray([box]), p_boxes)[0]
                idx = int(np.argmax(ious))
                if ious[idx] >= 0.5:
                    id_by_det_idx[idx] = tid

        # 3. Phone detection inside crops
        phones = []
        if crops:
            res_ph = self.phone_detector.predict(
                crops, classes=[67], conf=self.conf, iou=self.iou_th,
                device=self.device, verbose=False
            )
            for p_idx, (r, (ox, oy)) in enumerate(zip(res_ph, offs)):
                tid = id_by_det_idx.get(p_idx)
                for b in r.boxes:
                    px1, py1, px2, py2 = b.xyxy[0].cpu().numpy().astype(int)
                    s = float(b.conf[0])
                    phones.append((px1 + ox, py1 + oy, px2 + ox, py2 + oy, s, tid))

        # 4. Memory update of phone usage
        phone_owner_ids = {ph[-1] for ph in phones if ph[-1] is not None}
        for tid in phone_owner_ids:
            box = track_boxes.get(tid)
            if box is None:
                continue
            found = False
            for mem in self._mem:
                if mem["id"] == tid:
                    mem["bbox"], mem["ts"], found = box, now, True
                    break
            if not found:
                self._mem.append({"id": tid, "bbox": box, "ts": now})

        for mem in self._mem:
            tid = mem["id"]
            box = track_boxes.get(tid)
            if box is not None:
                mem["bbox"] = box

        self._mem = [m for m in self._mem if now - m["ts"] <= self.retention_secs]

        # 5. Drawing annotations
        annotated = None
        if draw:
            canvas = frame.copy()
            for mem in self._mem:
                tid = mem["id"]
                box = mem["bbox"]
                x1, y1, x2, y2 = box
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(canvas, f"ID {tid} | Person", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            annotated = canvas

        tracked_persons = [(box[0], box[1], box[2], box[3], tid)
                           for tid, box in track_boxes.items()]
        return tracked_persons, phones, (crops if return_crops else None), annotated

    # ------------------------- video util -----------------------------------
    def run_video(self, source: str | int = 0, *, show_fps: bool = True,
                  save_path: str | None = None, show_streaming: bool = True) -> None:
        TITLE = "TonAI Computer Vision"
        if 'rtsp://' in str(source):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;60000000"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 0)
        else:
            cap = cv2.VideoCapture(source)

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
    parser.add_argument("--phone_model_weights", type=str, default="weights/yolo12l.pt")
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
