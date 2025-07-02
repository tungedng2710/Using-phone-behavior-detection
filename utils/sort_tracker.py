import numpy as np
from filterpy.kalman import KalmanFilter
from utils.utils import iou_matrix, greedy_match


class _KalmanBox:
    _count = 0

    def __init__(self, bbox):
        # state: cx, cy, w, h, vx, vy, vw, vh
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i+4] = dt
        self.kf.H = np.eye(4, 8)
        self.kf.P *= 10.0
        self.kf.R *= 1.0
        self.kf.Q *= 0.01

        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)
        self.kf.x[:4, 0] = [cx, cy, w, h]

        self.time_since_update = 0
        self.hits = 1
        self.id = _KalmanBox._count
        _KalmanBox._count += 1

    @staticmethod
    def _xyxy_to_cxcywh(b):
        x1, y1, x2, y2 = b
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return cx, cy, w, h

    @staticmethod
    def _cxcywh_to_xyxy(c):
        cx, cy, w, h = c
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2])

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self._cxcywh_to_xyxy(self.kf.x[:4, 0])

    def update(self, bbox):
        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)
        self.kf.update(np.array([cx, cy, w, h]))
        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        return self._cxcywh_to_xyxy(self.kf.x[:4, 0])


class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets: np.ndarray):
        """Update tracker with detections.

        Args:
            dets: array Nx5 of [x1,y1,x2,y2,score]
        Returns:
            array of tracks [x1,y1,x2,y2,track_id]
        """
        # Predict existing trackers
        trks = []
        to_del = []
        for i, t in enumerate(self.trackers):
            box = t.predict()
            if np.any(np.isnan(box)):
                to_del.append(i)
                continue
            trks.append(box)
        for i in reversed(to_del):
            self.trackers.pop(i)
        trks = np.array(trks)

        if len(dets) == 0:
            matches = []
            unmatched_dets = []
            unmatched_trks = list(range(len(self.trackers)))
        else:
            if len(trks) == 0:
                matches = []
                unmatched_dets = list(range(len(dets)))
                unmatched_trks = []
            else:
                ious = iou_matrix(trks, dets[:, :4])
                cost = 1.0 - ious
                if 'linear_sum_assignment' in globals().keys():
                    row_idx, col_idx = linear_sum_assignment(cost)
                    matches = [(r, c) for r, c in zip(row_idx, col_idx) if ious[r, c] >= self.iou_threshold]
                else:
                    matches = greedy_match(cost)
                    matches = [(r, c) for r, c in matches if ious[r, c] >= self.iou_threshold]
                matched_trks = {m[0] for m in matches}
                matched_dets = {m[1] for m in matches}
                unmatched_trks = [i for i in range(len(self.trackers)) if i not in matched_trks]
                unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]

        # Update matched trackers with assigned detections
        for t_idx, d_idx in matches:
            self.trackers[t_idx].update(dets[d_idx, :4])

        # Create new trackers for unmatched detections
        for idx in unmatched_dets:
            self.trackers.append(_KalmanBox(dets[idx, :4]))

        # Prepare result
        ret = []
        new_trackers = []
        for t in self.trackers:
            if t.time_since_update < self.max_age and (t.hits >= self.min_hits):
                ret.append(np.append(t.get_state(), t.id))
            if t.time_since_update < self.max_age:
                new_trackers.append(t)
        self.trackers = new_trackers
        return np.array(ret)
