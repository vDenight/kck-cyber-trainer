import cv2

from .pose_letters import classify_pose
from .pose_service import overlay_result


class CameraService:
    """Obsluga kamerki laptopa: podglad + rozpoznawanie liter na kazdej klatce."""

    WINDOW_NAME = "KCK Cyber Trainer - kamera"
    EXIT_KEYS = (ord("q"), 27)  # 'q' lub ESC

    def __init__(self, camera_index: int = 0, pose_service=None, stabilizer=None,
                 mirror: bool = False):
        """
        pose_service: obiekt z metoda detect(frame) -> landmarks | None
        stabilizer: obiekt z metoda update(state) -> state (wygladzanie wyniku)
        mirror: jesli True, podglad jest lustrzany. Klasyfikacja dziala zawsze
                na surowej klatce (jednolity uklad wspolrzednych).
        """
        self.camera_index = camera_index
        self.pose_service = pose_service
        self.stabilizer = stabilizer
        self.mirror = mirror

    def run(self) -> None:
        """Otwiera kamere i pokazuje podglad z rozpoznana litera, az do q/ESC."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Nie moge otworzyc kamery. Sprawdz czy nie jest uzywana przez inny program.")
            return

        print("Kamera wlaczona. Nacisnij 'q' lub ESC w oknie podgladu aby wyjsc.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Nie udalo sie odczytac klatki z kamery.")
                    break

                view = self._process_frame(frame)
                if self.mirror:
                    view = cv2.flip(view, 1)
                cv2.imshow(self.WINDOW_NAME, view)

                if self._exit_requested():
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame):
        if self.pose_service is None:
            return frame

        landmarks = self.pose_service.detect(frame)
        state = classify_pose(landmarks)
        if self.stabilizer is not None:
            state = self.stabilizer.update(state)
        return overlay_result(frame, landmarks, state)

    def _exit_requested(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        return key in self.EXIT_KEYS