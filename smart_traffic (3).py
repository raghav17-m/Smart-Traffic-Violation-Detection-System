"""
Smart Traffic Violation Detection System
=========================================
Features:
  - Vehicle Detection (YOLOv8/YOLOv11 via ultralytics)
  - Vehicle Tracking (ByteTrack / BoT-SORT via supervision)
  - Vehicle Counting
  - Triple-Riding Detection on Motorcycles
  - Speed Tracking (pixel-based estimation with calibration line)
  - Speed Violation Detection (configurable speed limit)
  - Violation Screenshot Capture
  - CSV Violation Logging
  - Full OpenCV GUI Dashboard
  - PDF Report Generator (auto-generated at session end)

Requirements:
    pip install ultralytics opencv-python supervision pandas numpy reportlab

Usage:
    python smart_traffic.py                        # opens webcam
    python smart_traffic.py --source video.mp4     # use video file
    python smart_traffic.py --source video.mp4 --model yolov8n.pt
    python smart_traffic.py --source video.mp4 --speed-limit 60
    python smart_traffic.py --source video.mp4 --pixels-per-meter 8.5
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy imports – give clear error messages if missing
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("❌  ultralytics not found.  Run:  pip install ultralytics")

try:
    import supervision as sv
except ImportError:
    sys.exit("❌  supervision not found.  Run:  pip install supervision")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
except ImportError:
    sys.exit("❌  reportlab not found.  Run:  pip install reportlab")


# ===========================================================================
# CONSTANTS & PATHS
# ===========================================================================
BASE_DIR       = Path("SmartTraffic")
VIOLATIONS_DIR = BASE_DIR / "violations"
LOGS_DIR       = BASE_DIR / "logs"
MODELS_DIR     = BASE_DIR / "models"
REPORTS_DIR    = BASE_DIR / "reports"

for d in (VIOLATIONS_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CSV_LOG_PATH = LOGS_DIR / "violations.csv"

# YOLO class IDs (COCO)
CLASS_PERSON     = 0
CLASS_BICYCLE    = 1
CLASS_CAR        = 2
CLASS_MOTORCYCLE = 3
CLASS_BUS        = 5
CLASS_TRUCK      = 7

VEHICLE_CLASSES  = {CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK, CLASS_BICYCLE}
MOTORCYCLE_CLASS = {CLASS_MOTORCYCLE}

CLASS_NAMES = {
    CLASS_BICYCLE:    "Bicycle",
    CLASS_CAR:        "Car",
    CLASS_MOTORCYCLE: "Motorcycle",
    CLASS_BUS:        "Bus",
    CLASS_TRUCK:      "Truck",
}

TRIPLE_RIDING_THRESHOLD = 3   # persons on one motorcycle = violation

# Speed tracking defaults
DEFAULT_SPEED_LIMIT    = 60   # km/h
DEFAULT_PIXELS_PER_M   = 8.0  # pixels per metre (calibrate per camera setup)

# ---------------------------------------------------------------------------
# GUI Layout constants
# ---------------------------------------------------------------------------
PANEL_W      = 320          # right-side dashboard width
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD    = cv2.FONT_HERSHEY_DUPLEX

# Palette
C_BG         = (18,  18,  18)
C_ACCENT     = (0,  200, 120)
C_DANGER     = (0,   60, 220)
C_WARNING    = (0,  165, 255)
C_TEXT       = (230, 230, 230)
C_SUBTEXT    = (140, 140, 140)
C_MOTO_BOX   = (0,  200, 120)
C_PERSON_BOX = (255, 180,  50)
C_VEH_BOX    = (200, 200, 200)
C_VIOLATION  = (0,   60, 220)
C_SPEED_OK   = (0,  200, 120)
C_SPEED_WARN = (0,  165, 255)
C_SPEED_VIOL = (0,   60, 220)

BAR_H        = 50            # top HUD bar height


# ===========================================================================
# CSV LOGGER
# ===========================================================================
class ViolationLogger:
    def __init__(self, path: Path):
        self.path = path
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "track_id", "violation_type",
                    "rider_count", "speed_kmh", "speed_limit_kmh",
                    "vehicle_class", "screenshot_path"
                ])

    def log(self, track_id: int, violation_type: str,
            rider_count: int, screenshot_path: str,
            speed_kmh: float = 0.0, speed_limit: float = 0.0,
            vehicle_class: str = "Unknown"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ts, track_id, violation_type, rider_count,
                f"{speed_kmh:.1f}", f"{speed_limit:.1f}",
                vehicle_class, screenshot_path
            ])


# ===========================================================================
# SPEED TRACKER
# ===========================================================================
class SpeedTracker:
    """
    Estimates vehicle speed from centroid displacement between frames.

    pixels_per_meter : calibration factor (measure a known real-world distance
                       in the scene and count pixels).
    source_fps       : video FPS used to convert pixels/frame to km/h.
    smoothing        : number of frames to average speed over.
    """

    def __init__(self, pixels_per_meter: float = DEFAULT_PIXELS_PER_M,
                 source_fps: float = 30.0, smoothing: int = 10):
        self.ppm        = pixels_per_meter
        self.fps        = source_fps
        self.smoothing  = smoothing

        # {track_id: deque of centroids [(x, y), ...]}
        self.history: dict = defaultdict(list)
        # {track_id: current smoothed speed (km/h)}
        self.speeds:  dict = {}

    def update(self, track_id: int, box: np.ndarray) -> float:
        """
        Update centroid history for track_id and return estimated speed (km/h).
        box: [x1, y1, x2, y2]
        """
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        centroid = (cx, cy)

        self.history[track_id].append(centroid)
        if len(self.history[track_id]) > self.smoothing + 1:
            self.history[track_id].pop(0)

        hist = self.history[track_id]
        if len(hist) < 2:
            self.speeds[track_id] = 0.0
            return 0.0

        # Average displacement over the stored window
        displacements = []
        for i in range(1, len(hist)):
            dx = hist[i][0] - hist[i-1][0]
            dy = hist[i][1] - hist[i-1][1]
            displacements.append(np.sqrt(dx*dx + dy*dy))

        avg_disp_px  = np.mean(displacements)          # pixels / frame
        speed_mps    = (avg_disp_px / self.ppm) * self.fps   # m/s
        speed_kmh    = speed_mps * 3.6

        self.speeds[track_id] = round(speed_kmh, 1)
        return self.speeds[track_id]

    def get_speed(self, track_id: int) -> float:
        return self.speeds.get(track_id, 0.0)

    def prune(self, active_ids: set):
        """Remove stale track IDs to prevent memory leak."""
        stale = [tid for tid in list(self.history.keys()) if tid not in active_ids]
        for tid in stale:
            self.history.pop(tid, None)
            self.speeds.pop(tid, None)


# ===========================================================================
# ASSOCIATION HELPER  (IoU-based person ↔ motorcycle linking)
# ===========================================================================
def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / float(area_a + area_b - inter)


def expand_box(box, pct=0.20):
    x1, y1, x2, y2 = box
    dw = (x2 - x1) * pct
    dh = (y2 - y1) * pct
    return (x1 - dw, y1 - dh, x2 + dw, y2 + dh)


def person_on_motorcycle(person_box, moto_box, iou_thresh=0.05):
    expanded_moto = expand_box(moto_box, pct=0.30)
    return iou(person_box, expanded_moto) >= iou_thresh


# ===========================================================================
# PDF REPORT GENERATOR
# ===========================================================================
class PDFReportGenerator:
    """Generates a professional PDF report from the session data."""

    def __init__(self, output_path: Path):
        self.output_path = output_path

    def generate(self, session_data: dict, violations_df: pd.DataFrame):
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            spaceAfter=6,
            textColor=colors.HexColor("#00C878"),
            alignment=TA_CENTER,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#888888"),
            alignment=TA_CENTER,
            spaceAfter=16,
        )
        section_style = ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#00C878"),
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        )
        normal_style = ParagraphStyle(
            "NormalCustom",
            parent=styles["Normal"],
            fontSize=9,
            leading=14,
        )
        danger_style = ParagraphStyle(
            "Danger",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#DC3C00"),
        )

        story = []

        # ---- Header --------------------------------------------------------
        story.append(Paragraph("SMART TRAFFIC VIOLATION DETECTION", title_style))
        story.append(Paragraph("Session Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#00C878"), spaceAfter=12))

        # ---- Session Meta --------------------------------------------------
        story.append(Paragraph("Session Information", section_style))
        meta_data = [
            ["Report Generated",   datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Session Start",      session_data.get("session_start", "N/A")],
            ["Session End",        session_data.get("session_end",   "N/A")],
            ["Source",             str(session_data.get("source",   "N/A"))],
            ["Model",              str(session_data.get("model",    "N/A"))],
            ["Confidence Threshold", str(session_data.get("conf",  "N/A"))],
            ["Speed Limit (km/h)", str(session_data.get("speed_limit", DEFAULT_SPEED_LIMIT))],
            ["Pixels per Metre",   str(session_data.get("pixels_per_meter", DEFAULT_PIXELS_PER_M))],
        ]
        meta_table = Table(meta_data, colWidths=[5.5*cm, 10*cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#1A1A2E")),
            ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#00C878")),
            ("TEXTCOLOR",   (1, 0), (1, -1), colors.HexColor("#333333")),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#F7F7F7"), colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # ---- Summary Statistics -------------------------------------------
        story.append(Paragraph("Session Statistics", section_style))

        total_violations   = session_data.get("total_violations", 0)
        triple_violations  = session_data.get("triple_violations", 0)
        speed_violations   = session_data.get("speed_violations",  0)
        total_vehicles     = session_data.get("total_vehicles",    0)
        frames_processed   = session_data.get("frames_processed",  0)
        avg_fps            = session_data.get("avg_fps",           0.0)
        max_speed          = session_data.get("max_speed_recorded",0.0)
        avg_speed          = session_data.get("avg_speed_recorded",0.0)

        stats_data = [
            ["Metric", "Value"],
            ["Total Frames Processed",   str(frames_processed)],
            ["Total Vehicles Counted",   str(total_vehicles)],
            ["Average Processing FPS",   f"{avg_fps:.1f}"],
            ["", ""],
            ["Total Violations Logged",  str(total_violations)],
            ["  Triple-Riding Violations", str(triple_violations)],
            ["  Speed Violations",        str(speed_violations)],
            ["", ""],
            ["Max Speed Recorded (km/h)", f"{max_speed:.1f}"],
            ["Avg Speed Recorded (km/h)", f"{avg_speed:.1f}"],
        ]
        stats_table = Table(stats_data, colWidths=[9*cm, 6.5*cm])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A1A2E")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#00C878")),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F7F7F7"), colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            # Highlight violation rows
            ("TEXTCOLOR",   (0, 5), (-1, 7), colors.HexColor("#CC2200")),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 10))

        # ---- Vehicle Type Breakdown ----------------------------------------
        if not violations_df.empty and "vehicle_class" in violations_df.columns:
            story.append(Paragraph("Violations by Vehicle Type", section_style))
            vc_counts = violations_df["vehicle_class"].value_counts().reset_index()
            vc_counts.columns = ["Vehicle Class", "Count"]
            vc_data = [["Vehicle Class", "Violation Count"]] + vc_counts.values.tolist()
            vc_table = Table(vc_data, colWidths=[8*cm, 7.5*cm])
            vc_table.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A1A2E")),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#00C878")),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#F7F7F7"), colors.white]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING",   (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ]))
            story.append(vc_table)
            story.append(Spacer(1, 10))

        # ---- Detailed Violation Log ----------------------------------------
        if not violations_df.empty:
            story.append(PageBreak())
            story.append(Paragraph("Detailed Violation Log", section_style))
            story.append(Paragraph(
                f"Showing all {len(violations_df)} recorded violation(s).",
                normal_style
            ))
            story.append(Spacer(1, 8))

            display_cols = ["timestamp", "track_id", "violation_type",
                            "rider_count", "speed_kmh", "speed_limit_kmh", "vehicle_class"]
            display_cols = [c for c in display_cols if c in violations_df.columns]

            col_headers = {
                "timestamp":       "Timestamp",
                "track_id":        "Track ID",
                "violation_type":  "Violation",
                "rider_count":     "Riders",
                "speed_kmh":       "Speed (km/h)",
                "speed_limit_kmh": "Limit (km/h)",
                "vehicle_class":   "Vehicle",
            }
            header_row = [col_headers.get(c, c) for c in display_cols]
            table_data = [header_row]

            for _, row in violations_df.iterrows():
                table_data.append([str(row.get(c, "")) for c in display_cols])

            n_cols   = len(display_cols)
            col_w    = 15.5 * cm / n_cols
            col_widths = [col_w] * n_cols

            viol_table = Table(table_data, colWidths=col_widths, repeatRows=1)
            row_styles = [
                ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A1A2E")),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#00C878")),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#FFF5F5"), colors.white]),
            ]

            # Highlight speed violations in orange
            if "violation_type" in display_cols:
                vt_idx = display_cols.index("violation_type")
                for r_idx, row in enumerate(table_data[1:], start=1):
                    if "Speed" in row[vt_idx]:
                        row_styles.append(
                            ("BACKGROUND", (0, r_idx), (-1, r_idx),
                             colors.HexColor("#FFF3E0"))
                        )
                    if "Triple" in row[vt_idx]:
                        row_styles.append(
                            ("BACKGROUND", (0, r_idx), (-1, r_idx),
                             colors.HexColor("#FFEBEE"))
                        )

            viol_table.setStyle(TableStyle(row_styles))
            story.append(viol_table)

        # ---- Footer --------------------------------------------------------
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#CCCCCC")))
        story.append(Paragraph(
            "Generated by Smart Traffic Violation Detection System  |  "
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=7, textColor=colors.HexColor("#999999"),
                           alignment=TA_CENTER, spaceBefore=4)
        ))

        doc.build(story)
        print(f"[INFO] PDF report saved → {self.output_path}")


# ===========================================================================
# MAIN DETECTOR CLASS
# ===========================================================================
class SmartTrafficDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35,
                 speed_limit: float = DEFAULT_SPEED_LIMIT,
                 pixels_per_meter: float = DEFAULT_PIXELS_PER_M,
                 source_fps: float = 30.0):
        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf  = conf
        self.logger = ViolationLogger(CSV_LOG_PATH)

        # Supervision tracker (ByteTrack)
        self.tracker = sv.ByteTrack()

        # Speed tracking
        self.speed_limit   = speed_limit
        self.speed_tracker = SpeedTracker(
            pixels_per_meter=pixels_per_meter,
            source_fps=source_fps,
            smoothing=8,
        )

        # State
        self.total_vehicles   : int   = 0
        self.total_violations : int   = 0
        self.triple_violations: int   = 0
        self.speed_violations : int   = 0
        self.vehicle_ids_seen : set   = set()

        # Speed statistics
        self.all_speeds: list = []         # every non-zero speed reading
        self.max_speed_recorded: float = 0.0

        # Per-track violation cooldown  {track_id: last_violation_frame}
        self.violation_cooldown: dict = {}
        self.speed_viol_cooldown: dict = {}
        self.COOLDOWN_FRAMES = 60

        # Recent violations for dashboard (last 5)
        self.recent_violations: list = []

        # FPS tracking
        self._fps_t0       = time.time()
        self._fps_cnt      = 0
        self.fps_live      = 0.0
        self._fps_samples: list = []

        self.frame_idx  = 0
        self.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    def _update_fps(self):
        self._fps_cnt += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self.fps_live = self._fps_cnt / elapsed
            self._fps_samples.append(self.fps_live)
            self._fps_cnt = 0
            self._fps_t0  = time.time()

    def avg_fps(self) -> float:
        return float(np.mean(self._fps_samples)) if self._fps_samples else 0.0

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_idx += 1
        self._update_fps()

        h, w = frame.shape[:2]

        # ---- Run YOLO inference ----------------------------------------
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        boxes_xyxy  = results.boxes.xyxy.cpu().numpy()
        class_ids   = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        if len(boxes_xyxy) == 0:
            canvas = self._build_canvas(frame, [], [], [], [], {})
            return canvas

        # ---- Filter to vehicles + persons ------------------------------
        vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASSES))
        person_mask  = (class_ids == CLASS_PERSON)

        if vehicle_mask.sum() > 0:
            veh_det = sv.Detections(
                xyxy       = boxes_xyxy[vehicle_mask],
                class_id   = class_ids[vehicle_mask],
                confidence = confidences[vehicle_mask],
            )
            veh_det   = self.tracker.update_with_detections(veh_det)
            track_ids = veh_det.tracker_id if veh_det.tracker_id is not None else np.array([])
            veh_boxes = veh_det.xyxy
            veh_cls   = veh_det.class_id
        else:
            veh_boxes = np.empty((0, 4))
            veh_cls   = np.array([])
            track_ids = np.array([])

        # Motorcycle subset
        moto_indices = np.where(veh_cls == CLASS_MOTORCYCLE)[0]
        moto_boxes   = veh_boxes[moto_indices]
        moto_tids    = track_ids[moto_indices] if len(track_ids) else np.array([])

        # Person boxes (raw, no tracking needed)
        person_boxes = boxes_xyxy[person_mask]

        # ---- Vehicle counting ------------------------------------------
        for tid in track_ids:
            if tid not in self.vehicle_ids_seen:
                self.vehicle_ids_seen.add(tid)
                self.total_vehicles += 1

        # ---- Speed tracking for all vehicles ---------------------------
        speed_map: dict = {}   # {track_id: speed_kmh}
        active_ids = set(int(t) for t in track_ids)
        for box, tid in zip(veh_boxes, track_ids):
            spd = self.speed_tracker.update(int(tid), box)
            speed_map[int(tid)] = spd
            if spd > 0:
                self.all_speeds.append(spd)
                if spd > self.max_speed_recorded:
                    self.max_speed_recorded = spd
        self.speed_tracker.prune(active_ids)

        # ---- Speed violation detection ---------------------------------
        speed_violations_frame = []
        for box, cls_id, tid in zip(veh_boxes, veh_cls, track_ids):
            spd = speed_map.get(int(tid), 0.0)
            if spd > self.speed_limit:
                last_sv = self.speed_viol_cooldown.get(int(tid), -9999)
                if (self.frame_idx - last_sv) >= self.COOLDOWN_FRAMES:
                    self.speed_viol_cooldown[int(tid)] = self.frame_idx
                    self.total_violations += 1
                    self.speed_violations += 1

                    x1, y1, x2, y2 = map(int, box)
                    pad = 30
                    rx1 = max(0, x1-pad); ry1 = max(0, y1-pad)
                    rx2 = min(w, x2+pad); ry2 = min(h, y2+pad)
                    crop = frame[ry1:ry2, rx1:rx2].copy()
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    shot_path = str(VIOLATIONS_DIR / f"speed_{int(tid)}_{ts_str}.jpg")
                    cv2.imwrite(shot_path, crop)

                    vcls_name = CLASS_NAMES.get(int(cls_id), "Vehicle")
                    self.logger.log(
                        int(tid), "Speed-Violation", 0, shot_path,
                        speed_kmh=spd, speed_limit=self.speed_limit,
                        vehicle_class=vcls_name,
                    )
                    entry = {
                        "time"       : datetime.now().strftime("%H:%M:%S"),
                        "track_id"   : int(tid),
                        "type"       : "SPEED",
                        "detail"     : f"{spd:.0f} km/h (limit {self.speed_limit:.0f})",
                        "screenshot" : shot_path,
                    }
                    self.recent_violations.insert(0, entry)
                    self.recent_violations = self.recent_violations[:5]

                    speed_violations_frame.append((box, int(tid), spd))

        # ---- Triple-riding detection ------------------------------------
        violations_this_frame = []

        for mbox, tid in zip(moto_boxes, moto_tids):
            rider_count = sum(
                1 for pbox in person_boxes
                if person_on_motorcycle(pbox, mbox)
            )
            if rider_count >= TRIPLE_RIDING_THRESHOLD:
                last_viol = self.violation_cooldown.get(int(tid), -9999)
                if (self.frame_idx - last_viol) >= self.COOLDOWN_FRAMES:
                    self.violation_cooldown[int(tid)] = self.frame_idx
                    self.total_violations += 1
                    self.triple_violations += 1

                    x1, y1, x2, y2 = map(int, mbox)
                    pad = 40
                    rx1 = max(0, x1-pad); ry1 = max(0, y1-pad)
                    rx2 = min(w, x2+pad); ry2 = min(h, y2+pad)
                    crop = frame[ry1:ry2, rx1:rx2].copy()
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    shot_path = str(VIOLATIONS_DIR / f"triple_{int(tid)}_{ts_str}.jpg")
                    cv2.imwrite(shot_path, crop)

                    self.logger.log(
                        int(tid), "Triple-Riding", rider_count, shot_path,
                        speed_kmh=speed_map.get(int(tid), 0.0),
                        speed_limit=self.speed_limit,
                        vehicle_class="Motorcycle",
                    )
                    entry = {
                        "time"       : datetime.now().strftime("%H:%M:%S"),
                        "track_id"   : int(tid),
                        "type"       : "TRIPLE",
                        "detail"     : f"{rider_count} riders",
                        "screenshot" : shot_path,
                    }
                    self.recent_violations.insert(0, entry)
                    self.recent_violations = self.recent_violations[:5]

                violations_this_frame.append((mbox, int(tid), rider_count))

        # ---- Build annotated canvas ------------------------------------
        canvas = self._build_canvas(
            frame, veh_boxes, veh_cls, track_ids,
            violations_this_frame, speed_map,
            speed_violations_frame=speed_violations_frame,
        )
        return canvas

    # ------------------------------------------------------------------
    def _build_canvas(self, frame, veh_boxes, veh_cls, track_ids,
                      violations_this_frame, speed_map: dict = None,
                      speed_violations_frame: list = None):
        if speed_map is None:
            speed_map = {}
        if speed_violations_frame is None:
            speed_violations_frame = []

        h, w = frame.shape[:2]
        out_w = w + PANEL_W
        canvas = np.full((h + BAR_H, out_w, 3), C_BG, dtype=np.uint8)

        vis = frame.copy()
        violation_tids  = {v[1] for v in violations_this_frame}
        speed_viol_tids = {v[1] for v in speed_violations_frame}

        for box, cls_id, tid in zip(veh_boxes, veh_cls, track_ids):
            x1, y1, x2, y2 = map(int, box)
            tid_int    = int(tid)
            is_moto    = (cls_id == CLASS_MOTORCYCLE)
            is_triple  = (tid_int in violation_tids)
            is_speed_v = (tid_int in speed_viol_tids)
            spd        = speed_map.get(tid_int, 0.0)

            if is_triple or is_speed_v:
                color, thickness = C_VIOLATION, 3
            elif is_moto:
                color, thickness = C_MOTO_BOX, 2
            else:
                color, thickness = C_VEH_BOX, 1

            # Speed color indicator (border tint on top edge line)
            if spd > 0 and not (is_triple or is_speed_v):
                ratio = min(spd / self.speed_limit, 1.5)
                if ratio < 0.8:
                    spd_color = C_SPEED_OK
                elif ratio < 1.0:
                    spd_color = C_SPEED_WARN
                else:
                    spd_color = C_SPEED_VIOL
                cv2.line(vis, (x1, y1), (x2, y1), spd_color, 3)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            # Label
            if is_triple:
                v_data = next((v for v in violations_this_frame if v[1] == tid_int), None)
                label  = f"TRIPLE x{v_data[2]}  ID:{tid_int}" if v_data else f"TRIPLE ID:{tid_int}"
            elif is_speed_v:
                label = f"SPEED! {spd:.0f}km/h ID:{tid_int}"
            elif is_moto:
                label = f"Moto {spd:.0f}km/h ID:{tid_int}" if spd > 0 else f"Moto ID:{tid_int}"
            else:
                label = f"{spd:.0f}km/h ID:{tid_int}" if spd > 0 else f"ID:{tid_int}"

            (lw, lh), _ = cv2.getTextSize(label, FONT, 0.48, 1)
            ty = max(y1 - 4, lh + 2)
            cv2.rectangle(vis, (x1, ty - lh - 4), (x1 + lw + 4, ty + 2), color, -1)
            cv2.putText(vis, label, (x1 + 2, ty), FONT, 0.48,
                        (0, 0, 0), 1, cv2.LINE_AA)

        # Violation flash
        any_violation = bool(violations_this_frame or speed_violations_frame)
        if any_violation:
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), C_DANGER, -1)
            vis = cv2.addWeighted(vis, 0.88, overlay, 0.12, 0)
            msg = "! VIOLATION DETECTED !"
            cv2.putText(vis, msg, (10, h - 14),
                        FONT_BOLD, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        canvas[BAR_H:BAR_H + h, 0:w] = vis

        # ---- Top HUD bar -----------------------------------------------
        cv2.rectangle(canvas, (0, 0), (out_w, BAR_H), (28, 28, 28), -1)
        cv2.line(canvas, (0, BAR_H), (out_w, BAR_H), C_ACCENT, 1)
        cv2.putText(canvas, "SMART TRAFFIC VIOLATION DETECTION",
                    (12, 32), FONT_BOLD, 0.70, C_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {self.fps_live:.1f}",
                    (out_w - 130, 32), FONT, 0.60, C_SUBTEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, datetime.now().strftime("%H:%M:%S"),
                    (out_w - 260, 32), FONT, 0.60, C_SUBTEXT, 1, cv2.LINE_AA)

        # ---- Right dashboard panel ------------------------------------
        px = w + 12
        py = BAR_H + 18

        def panel_heading(text, y):
            cv2.putText(canvas, text, (px, y), FONT_BOLD, 0.55,
                        C_ACCENT, 1, cv2.LINE_AA)
            cv2.line(canvas, (px, y + 5), (w + PANEL_W - 12, y + 5), C_ACCENT, 1)
            return y + 22

        def stat_row(label, value, y, color=C_TEXT):
            cv2.putText(canvas, label,      (px,       y), FONT, 0.45, C_SUBTEXT, 1, cv2.LINE_AA)
            cv2.putText(canvas, str(value), (px + 170, y), FONT_BOLD, 0.55, color,    1, cv2.LINE_AA)
            return y + 22

        py = panel_heading("LIVE STATS", py)
        py = stat_row("Frame",          self.frame_idx,       py)
        py = stat_row("Total Vehicles", self.total_vehicles,  py)
        py = stat_row("Active Tracks",  len(track_ids),       py, C_ACCENT)
        moto_cnt = int((veh_cls == CLASS_MOTORCYCLE).sum()) if len(veh_cls) else 0
        py = stat_row("Motorcycles",    moto_cnt,             py, C_MOTO_BOX)
        py = stat_row("Violations",     self.total_violations, py,
                      C_DANGER if self.total_violations > 0 else C_TEXT)
        py += 6

        py = panel_heading("SPEED STATS", py)
        py = stat_row("Speed Limit",    f"{self.speed_limit:.0f} km/h", py)
        py = stat_row("Max Speed",
                      f"{self.max_speed_recorded:.0f} km/h",
                      py,
                      C_DANGER if self.max_speed_recorded > self.speed_limit else C_ACCENT)
        avg_spd = float(np.mean(self.all_speeds[-50:])) if self.all_speeds else 0.0
        py = stat_row("Recent Avg Spd", f"{avg_spd:.0f} km/h", py)
        py = stat_row("Speed Viols",    self.speed_violations, py,
                      C_DANGER if self.speed_violations > 0 else C_TEXT)
        py = stat_row("Triple Viols",   self.triple_violations, py,
                      C_DANGER if self.triple_violations > 0 else C_TEXT)
        py += 6

        py = panel_heading("DETECTION LEGEND", py)

        def legend_item(color, label, y):
            cv2.rectangle(canvas, (px, y - 10), (px + 16, y + 4), color, -1)
            cv2.putText(canvas, label, (px + 24, y), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)
            return y + 18

        py = legend_item(C_VEH_BOX,   "Vehicle",          py)
        py = legend_item(C_MOTO_BOX,  "Motorcycle",       py)
        py = legend_item(C_SPEED_OK,  "Speed OK",         py)
        py = legend_item(C_SPEED_WARN,"Speed Warning",    py)
        py = legend_item(C_VIOLATION, "Violation !!",     py)
        py += 6

        py = panel_heading("RECENT VIOLATIONS", py)
        if not self.recent_violations:
            cv2.putText(canvas, "No violations yet", (px, py),
                        FONT, 0.42, C_SUBTEXT, 1, cv2.LINE_AA)
            py += 18
        else:
            for v in self.recent_violations:
                if py + 56 > canvas.shape[0] - 10:
                    break
                card_x1, card_x2 = px - 6, w + PANEL_W - 8
                card_y1, card_y2 = py - 12, py + 50
                cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2),
                              (40, 20, 20), -1)
                cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2),
                              C_VIOLATION, 1)
                tag = "SPEED" if v["type"] == "SPEED" else "TRIPLE"
                cv2.putText(canvas, f"[{tag}] ID:{v['track_id']}  {v['detail']}",
                            (px, py + 2), FONT_BOLD, 0.46, C_DANGER, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"@ {v['time']}",
                            (px, py + 20), FONT, 0.40, C_WARNING, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"Saved: {Path(v['screenshot']).name[:32]}",
                            (px, py + 36), FONT, 0.36, C_SUBTEXT, 1, cv2.LINE_AA)
                py += 60

        footer_y = canvas.shape[0] - 16
        cv2.line(canvas, (w, footer_y - 22), (out_w, footer_y - 22), C_SUBTEXT, 1)
        cv2.putText(canvas, "Q/ESC: Quit   S: Screenshot   P: PDF Report",
                    (px, footer_y), FONT, 0.36, C_SUBTEXT, 1, cv2.LINE_AA)

        return canvas

    # ------------------------------------------------------------------
    def generate_pdf_report(self, session_data: dict) -> Path:
        ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path  = REPORTS_DIR / f"session_report_{ts_str}.pdf"

        violations_df = pd.DataFrame()
        if CSV_LOG_PATH.exists():
            try:
                violations_df = pd.read_csv(CSV_LOG_PATH)
            except Exception:
                pass

        session_data["max_speed_recorded"] = self.max_speed_recorded
        session_data["avg_speed_recorded"] = (
            float(np.mean(self.all_speeds)) if self.all_speeds else 0.0
        )
        session_data["total_vehicles"]    = self.total_vehicles
        session_data["total_violations"]  = self.total_violations
        session_data["triple_violations"] = self.triple_violations
        session_data["speed_violations"]  = self.speed_violations
        session_data["frames_processed"]  = self.frame_idx
        session_data["avg_fps"]           = self.avg_fps()
        session_data["session_start"]     = self.session_start
        session_data["session_end"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        gen = PDFReportGenerator(pdf_path)
        gen.generate(session_data, violations_df)
        return pdf_path


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Smart Traffic Violation Detector")
    ap.add_argument("--source",           default="0",
                    help="Video path or webcam index (default: 0)")
    ap.add_argument("--model",            default="yolov8n.pt",
                    help="YOLO model weights (default: yolov8n.pt)")
    ap.add_argument("--conf",             type=float, default=0.35,
                    help="Detection confidence threshold (default: 0.35)")
    ap.add_argument("--save-video",       action="store_true",
                    help="Save annotated output to SmartTraffic/output.mp4")
    ap.add_argument("--speed-limit",      type=float, default=DEFAULT_SPEED_LIMIT,
                    help=f"Speed limit in km/h (default: {DEFAULT_SPEED_LIMIT})")
    ap.add_argument("--pixels-per-meter", type=float, default=DEFAULT_PIXELS_PER_M,
                    help=f"Calibration: pixels per real-world metre (default: {DEFAULT_PIXELS_PER_M}). "
                         "Measure a known distance in the scene to calibrate.")
    ap.add_argument("--no-pdf",           action="store_true",
                    help="Skip automatic PDF report generation at session end")
    return ap.parse_args()


def main():
    args   = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open source: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Source        : {args.source}  ({fw}x{fh} @ {fps_src:.1f} fps)")
    print(f"[INFO] Speed limit   : {args.speed_limit} km/h")
    print(f"[INFO] Pixels/metre  : {args.pixels_per_meter}")
    print(f"[INFO] Violations    → {VIOLATIONS_DIR}")
    print(f"[INFO] CSV Log       → {CSV_LOG_PATH}")
    print(f"[INFO] PDF Reports   → {REPORTS_DIR}")
    print("[INFO] Press  Q / ESC  to quit  |  S  to screenshot  |  P  to generate PDF now")

    detector = SmartTrafficDetector(
        model_path       = args.model,
        conf             = args.conf,
        speed_limit      = args.speed_limit,
        pixels_per_meter = args.pixels_per_meter,
        source_fps       = fps_src,
    )

    writer = None
    if args.save_video:
        out_path = str(BASE_DIR / "output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps_src,
                                   (fw + PANEL_W, fh + BAR_H))
        print(f"[INFO] Saving video  → {out_path}")

    window_name = "Smart Traffic Detector  |  Q to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(fw + PANEL_W, 1400), min(fh + BAR_H, 820))

    session_data = {
        "source" : args.source,
        "model"  : args.model,
        "conf"   : args.conf,
        "speed_limit"      : args.speed_limit,
        "pixels_per_meter" : args.pixels_per_meter,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        canvas = detector.process_frame(frame)
        cv2.imshow(window_name, canvas)
        if writer is not None:
            writer.write(canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("[INFO] Quit by user.")
            break
        elif key in (ord("s"), ord("S")):
            ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = str(VIOLATIONS_DIR / f"manual_snap_{ts_str}.jpg")
            cv2.imwrite(snap_path, canvas)
            print(f"[INFO] Screenshot saved: {snap_path}")
        elif key in (ord("p"), ord("P")):
            pdf_path = detector.generate_pdf_report(dict(session_data))
            print(f"[INFO] PDF report generated: {pdf_path}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # ---- Summary -------------------------------------------------------
    print("\n" + "=" * 56)
    print("  SMART TRAFFIC — SESSION SUMMARY")
    print("=" * 56)
    print(f"  Total Frames Processed   : {detector.frame_idx}")
    print(f"  Total Vehicles Counted   : {detector.total_vehicles}")
    print(f"  Total Violations Logged  : {detector.total_violations}")
    print(f"    Triple-Riding          : {detector.triple_violations}")
    print(f"    Speed Violations       : {detector.speed_violations}")
    print(f"  Max Speed Recorded       : {detector.max_speed_recorded:.1f} km/h")
    avg_spd = float(np.mean(detector.all_speeds)) if detector.all_speeds else 0.0
    print(f"  Average Speed Recorded   : {avg_spd:.1f} km/h")
    print(f"  CSV Log                  : {CSV_LOG_PATH}")
    print(f"  Screenshots              : {VIOLATIONS_DIR}")
    print("=" * 56)

    if CSV_LOG_PATH.exists():
        df = pd.read_csv(CSV_LOG_PATH)
        if not df.empty:
            print("\n  Violation Log Preview (last 10):")
            print(df.tail(10).to_string(index=False))

    if not args.no_pdf:
        print("\n[INFO] Generating PDF report...")
        pdf_path = detector.generate_pdf_report(session_data)
        print(f"[INFO] PDF report ready: {pdf_path}")


if __name__ == "__main__":
    main()
