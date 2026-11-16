"""Serwis MediaPipe PoseLandmarker: wykrywanie punktow ciala i nakladanie wyniku na obraz."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Wyciszenie logow MediaPipe/LiteRT (musi byc ustawione PRZED importem mediapipe)
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .pose_letters import (
    LETTERS,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    _arm_geometry,
    shoulder_angle,
)

# Tymczasowe przelaczniki podgladu
SHOW_LETTERS = True    # wyswietlanie rozpoznanej litery (na srodku)
SHOW_ANGLES = True     # zaznaczanie katow + ich wartosci


@contextmanager
def _quiet_native_stderr():
    """Przekierowanie natywnego stderr (fd 2) na /dev/null - ukrywa logi LiteRT/MediaPipe.

    Pythonowy stderr jest odtwarzany po zakonczeniu bloku. Logi INFO/WARNING
    z C++ (environment.cc, model_load.cc itd.) ida na poziomie deskryptora,
    wiec zmienna srodowiskowa GLOG ich nie wycisza.
    """
    if not sys.stderr or not hasattr(sys.stderr, "fileno"):
        yield
        return
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
    except (OSError, ValueError):
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_lite.task"

# Pelny szkielet MediaPipe Pose (33 punkty) - jak POSE_CONNECTIONS w starym API
_POSE_CONNECTIONS = (
    # twarz
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (8, 9), (7, 9), (8, 10),
    # tors
    (11, 12), (11, 23), (12, 24), (23, 24),
    # lewa reka
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # prawa reka
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # lewa noga
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # prawa noga
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # uszy
    (10, 9), (8, 6), (3, 0),
)
_POSE_POINT_COUNT = 33

_COLOR_OK = (60, 200, 60)
_COLOR_STATUS = (40, 160, 255)
_COLOR_SKELETON = (200, 160, 40)
_COLOR_ANGLE = (255, 120, 255)
_COLOR_ANGLE_TEXT = (255, 255, 255)
_COLOR_TORSO = (120, 220, 220)

_ARC_RADIUS = 30  # promien luku kata przy lokciu (px)
_ANGLE_FONT_SCALE = 0.55

# Pary (bark,.lokiec, nadgarstek) dla obu ramion
_ARMS = (
    (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP),
)


def _sweep_deg(a1: float, a2: float) -> tuple[float, float]:
    """Krotki kierunek przejscia (start, krok) z a1 do a2, kazdy w stopniach."""
    diff = (a2 - a1) % 360.0
    if diff > 180.0:  # druga strona bedzie krotsza
        diff -= 360.0
    return a1, diff


def _put_text(frame, text, x, y, color, scale=_ANGLE_FONT_SCALE):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _draw_angle_arc(frame, vertex, p1, p2, text: str) -> None:
    """Rysuje luk kata przy punkcie `vertex` miedzy promieniami do p1 i p2 + tekst."""
    import math
    vx, vy = vertex
    a1 = math.degrees(math.atan2(p1[1] - vy, p1[0] - vx))
    a2 = math.degrees(math.atan2(p2[1] - vy, p2[0] - vx))
    start, diff = _sweep_deg(a1, a2)
    # punkty luku (co ~4 stopnie)
    steps = max(2, int(abs(diff) / 4))
    arc_pts = []
    for i in range(steps + 1):
        ang = math.radians(start + diff * i / steps)
        arc_pts.append((int(vx + _ARC_RADIUS * math.cos(ang)),
                        int(vy + _ARC_RADIUS * math.sin(ang))))
    for a, b in zip(arc_pts, arc_pts[1:]):
        cv2.line(frame, a, b, _COLOR_ANGLE, 2, cv2.LINE_AA)

    # etykieta na zewnatrz luku, w polowie kata
    mid = math.radians(start + diff / 2)
    lx = int(vx + (_ARC_RADIUS + 14) * math.cos(mid))
    ly = int(vy + (_ARC_RADIUS + 14) * math.sin(mid))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, _ANGLE_FONT_SCALE, 1)
    _put_text(frame, text, lx - tw // 2, ly + th // 2, _COLOR_ANGLE_TEXT)


class PoseService:
    """Wrapper na PoseLandmarker w trybie VIDEO (znaczniki czasu monotoniczne)."""

    def __init__(self, model_path: Path | str | None = None):
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Brak modelu: {self._model_path}\n"
                f"Pobierz go do tego folderu:\n  {MODEL_URL}"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        with _quiet_native_stderr():
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = int(time.time() * 1000)

    def detect(self, frame_bgr):
        """Zwraca liste landmarkow pierwszej wykrytej postaci albo None."""
        self._timestamp_ms += 33  # ~30 fps, monotonicznie rosnaco
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _quiet_native_stderr():
            result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def close(self) -> None:
        self._landmarker.close()


def overlay_result(frame, landmarks, label: str):
    """Rysuje caly szkielet ciala (33 punkty) oraz rozpoznana litere na srodku albo stan."""
    height, width = frame.shape[:2]

    if landmarks is not None:
        pts = {
            idx: (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
            for idx in range(min(_POSE_POINT_COUNT, len(landmarks)))
            if landmarks[idx].visibility > 0.5
        }
        for a, b in _POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], _COLOR_SKELETON, 2)
        for x, y in pts.values():
            cv2.circle(frame, (x, y), 4, _COLOR_SKELETON, -1)

        # licznik widocznych punktow - sprawdzamy maksimum sledzenia MediaPipe
        visible = len(pts)
        total = min(_POSE_POINT_COUNT, len(landmarks))
        info = f"punkty: {visible}/{total}"
        cv2.putText(frame, info, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, info, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _COLOR_STATUS, 2, cv2.LINE_AA)

        # zaznaczenie katow ramion (luk + wartosci)
        _overlay_angles(frame, landmarks, pts)

    if label in LETTERS and SHOW_LETTERS:
        # Duza litera na srodku ekranu
        text = label
        scale, thickness = 5.0, 12
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = (width - tw) // 2
        y = (height + th) // 2
        cv2.putText(frame, text, (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, _COLOR_OK, thickness, cv2.LINE_AA)
    elif label not in LETTERS:
        # Stan: "Nieznany gest" / "Brak postaci" / "Niewiarygodne punkty"
        cv2.putText(frame, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, _COLOR_STATUS, 2, cv2.LINE_AA)

    return frame


def _overlay_angles(frame, landmarks, pts) -> None:
    """Zaznacza katy na obrazie:
    - kat barku (lokiec->bark->tulow) przy kazdym barku - luk + wartosc,
    - kat w lokciu - luk + wartosc,
    - os tulowia (bark->biodro) jako linia odniesienia.
    """
    if landmarks is None or not SHOW_ANGLES:
        return

    # os tulowia: bark -> biodro (odniesienie dla katow barku)
    for sh_idx, hip_idx in ((LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)):
        if sh_idx in pts and hip_idx in pts:
            cv2.line(frame, pts[sh_idx], pts[hip_idx], _COLOR_TORSO, 2, cv2.LINE_AA)

    for sh_idx, e_idx, w_idx, hip_idx in _ARMS:
        if not all(landmarks[i].visibility > 0.5 for i in (sh_idx, e_idx, w_idx, hip_idx)):
            continue
        geom = _arm_geometry(landmarks[hip_idx], landmarks[sh_idx],
                             landmarks[e_idx], landmarks[w_idx])

        # kat barku: nadgarstek->bark->tulow (0 = reka w dol, 90 = poziomo, 180 = w gore)
        # dokladnie ten sam kat, ktory uzywaja reguly liter
        _draw_angle_arc(frame, pts[sh_idx], pts[w_idx], pts[hip_idx],
                        f"{shoulder_angle(landmarks[hip_idx], landmarks[sh_idx], landmarks[w_idx]):.0f}")

        # kat w lokciu
        _draw_angle_arc(frame, pts[e_idx], pts[sh_idx], pts[w_idx],
                        f"{geom.elbow_angle:.0f}")