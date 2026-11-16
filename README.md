# KCK Cyber Trainer

Interfejs gestowy: rozpoznawanie liter **T, Y, I, L** na podstawie punktów ciała
(MediaPipe PoseLandmarker) z podglądem na żywo z kamerki.

## Uruchomienie

```
uv sync
uv run kck-cyber-trainer
```

Wyjście: klawisz `q` lub `ESC` w oknie podglądu.

## Model

Program wymaga modelu `pose_landmarker_lite.task` w folderze `models/`:

```
mkdir models
curl -L -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Rozpoznawane litery

| Litera | Pozycja |
|--------|---------|
| **T** | Obie ręce wyprostowane i rozłożone poziomo na wysokości barków. |
| **Y** | Obie ręce wyprostowane, uniesione ukośnie w górę i rozchylone na boki. |
| **I** | Obie ręce wyprostowane pionowo nad głową. |
| **L** | Prawa ręka wyprostowana pionowo w górę, lewa wyprostowana poziomo w bok. |

Prawa/lewa strona to strony **osoby** (MediaPipe oznacza punkty względem ciała).
Klasyfikacja działa na surowej klatce w jednym, konsekwentnym układzie;
opcjonalny podgląd lustrzany (`mirror=True`) odwraca obraz dopiero przy wyświetlaniu.

## Stany programu

- rozpoznana litera — duża literka na środku ekranu,
- `Nieznany gest` — żadna z reguł nie pasuje,
- `Brak postaci` — MediaPipe nie wykrył osoby,
- `Niewiarygodne punkty` — barki/łokcie/nadgarstki słabo widoczne.

Wynik jest stabilizowany głosowaniem większości w ostatnich 5 klatkach
(litera musi pojawić się w ≥ 3 klatkach); stany `Brak postaci` i
`Niewiarygodne punkty` pokazywane są natychmiast.

## Struktura

```
src/kck_cyber_trainer/
├── main.py          # entry point (składa serwis + kamere)
├── camera.py        # CameraService: petla podgladu z kamery
├── pose_service.py  # PoseService (MediaPipe) + overlay_result (rysowanie)
└── pose_letters.py  # reguly geometryczne T/Y/I/L + LetterStabilizer
tests/test_pose_letters.py
```

## Testy

```
uv run python -m tests.test_pose_letters
```