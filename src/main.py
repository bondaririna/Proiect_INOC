import os

import cv2
import mediapipe as mp


def _read_int_env(name: str, default: int) -> int:
    # Citeste o valoare intreaga din variabilele de mediu
    # Daca lipseste sau este invalida, foloseste valoarea implicita
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _read_float_env(name: str, default: float) -> float:
    # Citeste o valoare reala (float) din variabilele de mediu
    # Daca lipseste sau este invalida, foloseste valoarea implicita
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def main() -> None:
    # Parametri de configurare pentru camera si detectie
    # Pot fi schimbati din fisierul .env
    camera_index = _read_int_env("CAMERA_INDEX", 0)
    frame_width = _read_int_env("FRAME_WIDTH", 1280)
    frame_height = _read_int_env("FRAME_HEIGHT", 720)
    min_detection_confidence = _read_float_env("MIN_DETECTION_CONFIDENCE", 0.5)
    min_tracking_confidence = _read_float_env("MIN_TRACKING_CONFIDENCE", 0.5)

    # Deschide camera web si seteaza rezolutia dorita
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Oprire controlata daca nu se poate deschide camera
    if not cap.isOpened():
        raise RuntimeError(
            "Camera could not be opened. Verify CAMERA_INDEX and camera permissions."
        )

    # Modulele principale din MediaPipe:
    # - Hands: detectie si tracking mana
    # - drawing_utils: desenarea punctelor si conexiunilor pe frame
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Configureaza detectorul de maini
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        # Bucla principala: citeste cadrele video pana la iesire
        while True:
            # Citeste un cadru din fluxul camerei
            ok, frame = cap.read()
            if not ok:
                # Daca nu primeste cadru valid, iese din bucla
                break

            # Oglindeste imaginea
            frame = cv2.flip(frame, 1)
            # MediaPipe proceseaza imaginea in format RGB, nu BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Daca sunt maini detectate deseneaza landmark-urile pe frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # Aici se poate integra logica de clasificare a gesturilor
            cv2.putText(
                frame,
                "setup running - press Q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            # Afiseaza fereastra cu rezultatul procesarii
            cv2.imshow("INOC Hands-Free Cooking Assistant", frame)

            # Iesire din aplicatie la apasarea tastei Q.
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break

    # Elibereaza camera si inchide toate ferestrele OpenCV
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
