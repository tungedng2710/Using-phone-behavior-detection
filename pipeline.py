import argparse
from ultralytics import YOLO
import cv2, numpy as np, time, math
try:
    from scipy.optimize import linear_sum_assignment
    _has_scipy = True
except ImportError:
    _has_scipy = False
from utils import iou_matrix, greedy_match


class MobilePhoneDetection:
    def __init__(self,
                 person_model_weights = "yolov8s.pt",
                 phone_model_weights = "yolov8s.pt",
                 conf_thresh=0.25,
                 iou_thresh=0.45,
                 device="cuda"):
        self.person_detector = YOLO(person_model_weights)
        self.phone_detector = YOLO(phone_model_weights)
        self.conf   = conf_thresh
        self.iou_th = iou_thresh
        self.device = device
        self._mem: list[dict] = []
        self.retention_secs = 5      


    def _match(self, persons, phones):
        """
        Returns list of tuples (p_idx, ph_idx) indicating phone assigned to person.
        Criterion: maximise IoU  (equiv. minimise -IoU).
        """
        if not persons or not phones:
            return []

        p_xyxy = np.array([p[:4] for p in persons])
        ph_xyxy = np.array([ph[:4] for ph in phones])
        ious = iou_matrix(p_xyxy, ph_xyxy)        # NxM

        # Convert to cost matrix (Hungarian minimizes)
        cost = 1.0 - ious
        if _has_scipy:
            row_idx, col_idx = linear_sum_assignment(cost)
            matches = [(r, c) for r, c in zip(row_idx, col_idx)
                       if ious[r, c] > 0.0]      # ignore zero overlap matches
        else:
            matches = greedy_match(cost)
            matches = [(r, c) for r, c in matches if ious[r, c] > 0.0]
        return matches
    
    # -------------------------- public predict API ---------------------------
    def predict(
            self,
            frame: np.ndarray,
            *,
            draw: bool = True,
            return_crops: bool = True
        ):
        """
        Detect persons → crop → detect phones.
        Draw persons that are (or were within retention_secs) using a phone.
        """
        now = time.time()

        # ------------------------------------------------------------------ #
        # 1. Person detection                                                #
        # ------------------------------------------------------------------ #
        res_p = self.person_detector.predict(
            frame, classes=[0], conf=self.conf, iou=self.iou_th, # class id 0: person
            device=self.device, verbose=False)[0]

        persons, crops, offs = [], [], []
        for b in res_p.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            s = float(b.conf[0])
            persons.append((x1, y1, x2, y2, s))

            if return_crops:
                x1c, y1c = max(x1, 0), max(y1, 0)
                x2c, y2c = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                crops.append(frame[y1c:y2c, x1c:x2c].copy())
                offs.append((x1c, y1c))

        # ------------------------------------------------------------------ #
        # 2. Phone detection inside crops                                    #
        # ------------------------------------------------------------------ #
        phones = []
        if crops:
            res_ph = self.phone_detector.predict(
                crops, classes=[67], conf=self.conf, iou=self.iou_th, # class id 67: cell phone
                device=self.device, verbose=False)

            for p_idx, (r, (ox, oy)) in enumerate(zip(res_ph, offs)):
                for b in r.boxes:
                    px1, py1, px2, py2 = b.xyxy[0].cpu().numpy().astype(int)
                    s = float(b.conf[0])
                    phones.append((px1+ox, py1+oy, px2+ox, py2+oy, s, p_idx))

        # ------------------------------------------------------------------ #
        # 3. Memory update                                                   #
        # ------------------------------------------------------------------ #
        phone_owner_idxs = {ph[-1] for ph in phones}

        for idx in phone_owner_idxs:                           # refresh / insert
            box = persons[idx][:4]
            matched = False
            for mem in self._mem:
                if iou_matrix(np.asarray([mem["bbox"]]), np.asarray([box]))[0, 0] >= 0.5:
                    mem["bbox"], mem["ts"], matched = box, now, True
                    break
            if not matched:
                self._mem.append({"bbox": box, "ts": now})

        # drop expired
        self._mem = [m for m in self._mem if now - m["ts"] <= self.retention_secs]

        # ------------------------------------------------------------------ #
        # 4. Drawing                                                         #
        # ------------------------------------------------------------------ #
        annotated = None
        if draw:
            canvas = frame.copy()

            # current phone users
            for idx in phone_owner_idxs:
                x1, y1, x2, y2, s = persons[idx]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, f"Person {s:.2f}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            for (x1, y1, x2, y2, s, p_idx) in phones:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(canvas, f"Phone {s:.2f}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                px1, py1, px2, py2, _ = persons[p_idx]
                cv2.line(canvas,
                        ((px1+px2)//2, (py1+py2)//2),
                        ((x1+x2)//2,  (y1+y2)//2),
                        (0, 255, 255), 1)

            # remembered persons (no phone this frame, still within 5 s)
            for mem in self._mem:
                box = mem["bbox"]
                if any(iou_matrix(np.asarray([box]), 
                                np.asarray([persons[i][:4]]))[0, 0] >= 0.9
                    for i in phone_owner_idxs):
                    continue
                x1, y1, x2, y2 = box
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 180, 0), 2)

            annotated = canvas

        return (
            persons,
            phones,
            crops if return_crops else None,
            annotated
        )

    # ------------------------------------------------------------------ #
    # 5. Live video streaming                                            #
    # ------------------------------------------------------------------ #
    def run_video(
            self,
            source: str | int = 0,
            *,
            show_fps: bool = True,
            save_path: str | None = None,
            show_streaming: bool = True
        ) -> None:
        """
        Stream inference on a webcam / file / RTSP URL.

        Args
        ----
        source          Path / URL / camera index (default 0 = first webcam).
        show_fps        Overlay instantaneous FPS on the frame.
        save_path       If given, writes annotated video to this path (codec mp4v).
        show_streaming  If False, skips cv2.imshow (useful for headless servers).
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Optional writer
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer  = cv2.VideoWriter(save_path, fourcc, fps_src, (w, h))

        try:
            print("Object detection is running...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                *_, annotated = self.predict(frame, draw=True)
                dt = time.time() - t0

                if show_fps:
                    fps = 1.0 / (dt + 1e-6)
                    cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if writer:
                    writer.write(annotated)

                if show_streaming:
                    cv2.imshow("TonAI Computer Vision", annotated)
                    # Quit on ESC
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
    pipe.run_video(source=args.source,
                   show_streaming=args.show_streaming,
                   save_path=args.save_path)