"""Reguly geometryczne rozpoznawania liter T, Y, I, L na podstawie punktow ciala.

Punkty pochodza z MediaPipe PoseLandmarker (wspolrzedne znormalizowane,
os Y obrazu wskazuje w dol). Klasyfikacja to projekcja na obraz -
prawa/lewa strona to strona OSOBY (MediaPipe oznacza je wg ciala).

Kat barku (nadgarstek->bark->tulow): kat przy barku miedzy cala reka
(bark->nadgarstek) a osia tulowia (bark->biodro, wskazuje w DOL).
    ~0°   = reka wzdloz ciala w dol
    ~90°  = reka poziomo na wysokosci barku (T, pozioma czesc L)
    ~180° = reka pionowo w gore (I, pionowa czesc L)
    Y wypada ok. 105°-145°.
Reguly sa wylacznie katowe - bez dodatkowych tolerancji wysokosci.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import atan2, degrees

# Indeksy punktow ciala MediaPipe Pose
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

# Rozpoznawane litery i stany
T, Y, I, L = "T", "Y", "I", "L"
LETTERS = (T, Y, I, L)
UNKNOWN = "Nieznany gest"
NO_PERSON = "Brak postaci"
UNRELIABLE = "Niewiarygodne punkty"


@dataclass(frozen=True)
class Tolerances:
    """Tolerancje regul geometrycznych (katy w stopniach).

    Kat barku = nadgarstek->bark->tulow (0 = reka w dol, 90 = poziomo, 180 = w gore).
    """

    min_elbow_angle: float = 150.0     # kat w lokciu od ktorego reka jest "prosta"
    # I: obie rece pionowo nad glowa - kat barku nie mniejszy niz:
    i_min_shoulder_angle: float = 160.0
    # Y: obie rece ukosnie w gore - kat barku w przedziale:
    y_min_shoulder_angle: float = 105.0
    y_max_shoulder_angle: float = 145.0
    # L: lewa reka poziomo w bok - kat barku w przedziale:
    l_min_left_shoulder_angle: float = 75.0
    l_max_left_shoulder_angle: float = 105.0
    # L: prawa reka pionowo w gore - kat barku nie mniejszy niz:
    l_min_right_shoulder_angle: float = 150.0
    # T: obie rece poziomo na wysokosci barkow - kat barku w przedziale:
    t_min_shoulder_angle: float = 75.0
    t_max_shoulder_angle: float = 105.0
    min_visibility: float = 0.5        # minimalna wiarygodnosc punktow uzywanych w regulach


DEFAULT = Tolerances()


@dataclass(frozen=True)
class ArmGeometry:
    """Parametry jednego ramienia w projekcji na obraz."""

    elbow_angle: float     # kat przy lokciu (180 = reka idealnie prosta)
    shoulder_angle: float  # kat barku nadgarstek->bark->tulow (0 w dol, 90 poziomo, 180 w gore)


def _angle_between(ax, ay, bx, by) -> float:
    """Kat miedzy dwoma wektorami, 0-180 stopni."""
    dot = ax * bx + ay * by
    ma = (ax * ax + ay * ay) ** 0.5
    mb = (bx * bx + by * by) ** 0.5
    if ma < 1e-9 or mb < 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (ma * mb)))
    return degrees(atan2((1.0 - cos * cos) ** 0.5, cos))


def shoulder_angle(hip, shoulder, wrist) -> float:
    """Kat barku: nadgarstek->bark->tulow (bark->biodro, os wskazuje w dol).

    0° = reka wzdloz ciala w dol, 90° = poziomo, 180° = pionowo w gore.
    """
    return _angle_between(wrist.x - shoulder.x, wrist.y - shoulder.y,
                           hip.x - shoulder.x, hip.y - shoulder.y)


def _elbow_angle(shoulder, elbow, wrist) -> float:
    """Kat miedzy ramieniem (lokiec->bark) i przedramieniem (lokiec->nadgarstek)."""
    return _angle_between(shoulder.x - elbow.x, shoulder.y - elbow.y,
                           wrist.x - elbow.x, wrist.y - elbow.y)


def _arm_geometry(hip, shoulder, elbow, wrist) -> ArmGeometry:
    return ArmGeometry(
        elbow_angle=_elbow_angle(shoulder, elbow, wrist),
        shoulder_angle=shoulder_angle(hip, shoulder, wrist),
    )


def _head_top_y(landmarks) -> float | None:
    """Y gory glowy - najwyzej polozony widoczny punkt glowy.

    None, gdy zaden punkt glowy nie jest widoczny (glowa poza kadrem) -
    wtedy litery I nie da sie zweryfikowac.
    """
    candidates = [
        landmarks[idx].y
        for idx in (LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
        if landmarks[idx].visibility > 0.5 and _in_frame(landmarks[idx])
    ]
    return min(candidates) if candidates else None


def _in_frame(lm) -> bool:
    """Czy punkt lezy w kadrze (z marginesem na szum przy krawedzi)."""
    return -0.05 <= lm.x <= 1.05 and -0.05 <= lm.y <= 1.05


def _unreliable(landmarks, indices, tol: Tolerances) -> bool:
    """Prawda, gdy ktorys z punktow jest poza kadrem lub niewidoczny.

    MediaPipe przy obcietej rece (nadgarstek/lokiec poza kadr) potrafi
    zwrocic punkt z dobra widocznoscia, ale lezacy poza obrazem -
    wtedy wspolrzedne sa ekstrapolowane i nie nadaja sie do regul.
    """
    return any(
        landmarks[i].visibility < tol.min_visibility or not _in_frame(landmarks[i])
        for i in indices
    )


def classify_pose(landmarks, tol: Tolerances = DEFAULT) -> str:
    """Przypisuje litere (T, Y, I, L) albo stan (UNKNOWN/NO_PERSON/UNRELIABLE).

    Kolejnosc sprawdzania regul: I -> L -> T -> Y (od najwiecej ograniczajacej).
    """
    if not landmarks:
        return NO_PERSON

    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
              LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP)
    if _unreliable(landmarks, needed, tol):
        return UNRELIABLE

    left = _arm_geometry(landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER],
                         landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
    right = _arm_geometry(landmarks[RIGHT_HIP], landmarks[RIGHT_SHOULDER],
                         landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])

    if not (left.elbow_angle >= tol.min_elbow_angle
            and right.elbow_angle >= tol.min_elbow_angle):
        return UNKNOWN

    lw, rw = landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST]
    head_top = _head_top_y(landmarks)

    # I: obie rece proste, pionowo nad glowa (kat barku bliski 180);
    # glowa musi byc widoczna, zeby mialo sens nadgarstek "nad glowa"
    if (head_top is not None
            and left.shoulder_angle >= tol.i_min_shoulder_angle
            and right.shoulder_angle >= tol.i_min_shoulder_angle
            and lw.y < head_top and rw.y < head_top):
        return I

    # L: prawa reka pionowo w gore (min), lewa poziomo w bok (przedzial)
    if (right.shoulder_angle >= tol.l_min_right_shoulder_angle
            and tol.l_min_left_shoulder_angle <= left.shoulder_angle <= tol.l_max_left_shoulder_angle):
        return L

    # T: obie rece poziomo na wysokosci barkow (przedzial)
    if (tol.t_min_shoulder_angle <= left.shoulder_angle <= tol.t_max_shoulder_angle
            and tol.t_min_shoulder_angle <= right.shoulder_angle <= tol.t_max_shoulder_angle):
        return T

    # Y: obie rece ukosnie w gore (przedzial)
    if (tol.y_min_shoulder_angle <= left.shoulder_angle <= tol.y_max_shoulder_angle
            and tol.y_min_shoulder_angle <= right.shoulder_angle <= tol.y_max_shoulder_angle):
        return Y

    return UNKNOWN


class LetterStabilizer:
    """Wygladzenie wyniku glosowaniem wiekszosci w ostatnich klatkach.

    Stan NO_PERSON / UNRELIABLE pokazywany od razu (bez opoznienia),
    zeby wynik nie "wisnal" po zniknieciu osoby z kadru.
    """

    def __init__(self, window: int = 5, min_votes: int | None = None):
        self._history: deque[str] = deque(maxlen=window)
        self._min_votes = min_votes if min_votes is not None else window // 2 + 1

    def update(self, state: str) -> str:
        if state in (NO_PERSON, UNRELIABLE):
            self._history.clear()
            return state
        self._history.append(state)
        letter, votes = Counter(self._history).most_common(1)[0]
        if letter in LETTERS and votes >= self._min_votes:
            return letter
        return state

    def reset(self) -> None:
        self._history.clear()