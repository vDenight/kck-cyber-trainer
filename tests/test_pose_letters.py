"""Testy regul klasyfikacji liter na syntetycznych punktach ciala."""

from types import SimpleNamespace as LM

from kck_cyber_trainer.pose_letters import (
    I, L, NO_PERSON, T, UNRELIABLE, UNKNOWN, Y, LetterStabilizer, classify_pose,
)


def pose(left_wrist, right_wrist, left_elbow, right_elbow,
         ls=(0.45, 0.60), rs=(0.55, 0.60),
         lh=(0.45, 0.80), rh=(0.55, 0.80),
         vis=1.0, eyes=0.30):
    def lm(x, y):
        return LM(x=x, y=y, visibility=vis)
    pts = [lm(0.5, 0.5) for _ in range(33)]
    # glowa
    pts[2], pts[5] = lm(0.47, eyes), lm(0.53, eyes)
    pts[7], pts[8] = lm(0.46, eyes + 0.02), lm(0.54, eyes + 0.02)
    # barki, biodra, lokcie, nadgarstki (strony ciala osoby)
    pts[11], pts[12] = lm(*ls), lm(*rs)
    pts[23], pts[24] = lm(*lh), lm(*rh)
    pts[13], pts[14] = lm(*left_elbow), lm(*right_elbow)
    pts[15], pts[16] = lm(*left_wrist), lm(*right_wrist)
    return pts


def test_T():
    assert classify_pose(pose((0.25, 0.60), (0.75, 0.60), (0.35, 0.60), (0.65, 0.60))) == T


def test_Y():
    assert classify_pose(pose((0.30, 0.45), (0.70, 0.45), (0.38, 0.53), (0.62, 0.53))) == Y


def test_I():
    assert classify_pose(pose((0.42, 0.20), (0.58, 0.20), (0.43, 0.40), (0.57, 0.40))) == I


def test_L():
    assert classify_pose(pose((0.25, 0.60), (0.55, 0.20), (0.35, 0.60), (0.55, 0.40))) == L


def test_L_rejected_when_left_arm_too_high():
    # L wymaga lewej poziomej (50-130 stopni); lewa ukosna w gore (~135 stopni) -> nie L
    assert classify_pose(pose((0.30, 0.45), (0.55, 0.20), (0.36, 0.52), (0.55, 0.40))) != L


def test_unknown_gesture():
    # rece opuszczone w dol
    assert classify_pose(pose((0.40, 0.80), (0.60, 0.80), (0.42, 0.70), (0.58, 0.70))) == UNKNOWN


def test_unreliable_points():
    p = pose((0.42, 0.20), (0.58, 0.20), (0.43, 0.40), (0.57, 0.40), vis=0.3)
    assert classify_pose(p) == UNRELIABLE


def test_unreliable_when_point_outside_frame():
    # MediaPipe potrafi zwrocic punkt z dobra widocznoscia poza kadrem
    # (obcieta reke) - wspolrzedne poza [0,1] sa ekstrapolowane.
    p = pose((0.42, 0.20), (0.58, 0.20), (0.43, 0.40), (0.57, 0.40))
    p[16] = LM(x=1.2, y=-0.3, visibility=1.0)  # prawy nadgarstek poza kadrem
    assert classify_pose(p) == UNRELIABLE


def test_I_not_matched_without_visible_head():
    # rece w gore, ale glowa poza kadrem -> I nie moze byc zweryfikowane
    p = pose((0.42, 0.20), (0.58, 0.20), (0.43, 0.40), (0.57, 0.40), eyes=0.9)
    for idx in (2, 5, 7, 8):  # oczy i uszy poza kadrem
        p[idx] = LM(x=0.5, y=1.4, visibility=0.9)
    assert classify_pose(p) != I


def test_no_person():
    assert classify_pose(None) == NO_PERSON


def test_stabilizer_resets_on_no_person():
    s = LetterStabilizer(window=5)
    for _ in range(5):
        assert s.update(T) == T
    # osoba znika -> natychmiast Brak postaci (nie "wisi" ostatnia litera)
    assert s.update(NO_PERSON) == NO_PERSON
    # po powrocie litera musi zdobyc wiekszosc again
    assert s.update(T) == T  # historia wyczyszczona, pierwsza klatka z litera = 1/1 wiekszosc


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")