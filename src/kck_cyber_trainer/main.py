from kck_cyber_trainer.camera import CameraService
from kck_cyber_trainer.pose_letters import LetterStabilizer
from kck_cyber_trainer.pose_service import PoseService


def main():
    try:
        pose = PoseService()
    except FileNotFoundError as exc:
        print(exc)
        return

    stabilizer = LetterStabilizer(window=5)
    try:
        CameraService(pose_service=pose, stabilizer=stabilizer).run()
    finally:
        pose.close()


if __name__ == "__main__":
    main()