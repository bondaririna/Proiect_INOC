import os
import time
from collections import deque
from typing import Deque, Tuple

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


def _send_right_arrow() -> bool:
    # Trimite global tasta Right Arrow pe Windows.
    # YouTube interpreteaza Right Arrow ca seek +5 secunde.
    if os.name != "nt":
        return False

    import ctypes

    vk_right = 0x27
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.keybd_event(vk_right, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_right, 0, keyeventf_keyup, 0)
    return True


def _send_left_arrow() -> bool:
    # Trimite global tasta Left Arrow pe Windows.
    # YouTube interpreteaza Left Arrow ca seek -5 secunde.
    if os.name != "nt":
        return False

    import ctypes

    vk_left = 0x25
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.keybd_event(vk_left, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_left, 0, keyeventf_keyup, 0)
    return True


def _send_k_key() -> bool:
    # Trimite global tasta K pe Windows.
    # YouTube interpreteaza K ca play/pause.
    if os.name != "nt":
        return False

    import ctypes

    vk_k = 0x4B
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.keybd_event(vk_k, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_k, 0, keyeventf_keyup, 0)
    return True


def _send_ctrl_w() -> bool:
    # Trimite global combinatia Ctrl+W pe Windows.
    # In browser, inchide tab-ul curent (de ex. YouTube).
    if os.name != "nt":
        return False

    import ctypes

    vk_control = 0x11
    vk_w = 0x57
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.keybd_event(vk_control, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_w, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_w, 0, keyeventf_keyup, 0)
    ctypes.windll.user32.keybd_event(vk_control, 0, keyeventf_keyup, 0)
    return True


def _send_volume_up() -> bool:
    # Trimite tasta media Volume Up pe Windows.
    if os.name != "nt":
        return False

    import ctypes

    vk_volume_up = 0xAF
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.keybd_event(vk_volume_up, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_volume_up, 0, keyeventf_keyup, 0)
    return True


def _send_volume_down() -> bool:
    # Trimite tasta media Volume Down pe Windows.
    if os.name != "nt":
        return False

    import ctypes

    vk_volume_down = 0xAE
    keyeventf_keyup = 0x0002
    ctypes.windll.user32.keybd_event(vk_volume_down, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_volume_down, 0, keyeventf_keyup, 0)
    return True


def _is_open_palm_pose(hand_landmarks, mp_hands, min_palm_span_x: float) -> bool:
    lm = hand_landmarks.landmark
    index_up = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[
        mp_hands.HandLandmark.INDEX_FINGER_PIP
    ].y
    middle_up = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP
    ].y
    ring_up = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[
        mp_hands.HandLandmark.RING_FINGER_PIP
    ].y
    pinky_up = lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[
        mp_hands.HandLandmark.PINKY_PIP
    ].y
    raised_count = int(index_up) + int(middle_up) + int(ring_up) + int(pinky_up)
    if raised_count < 3:
        return False

    palm_span_x = abs(
        lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        - lm[mp_hands.HandLandmark.PINKY_MCP].x
    )
    if palm_span_x < (min_palm_span_x * 0.75):
        return False

    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_open = (
        abs(thumb_tip.x - thumb_ip.x) > 0.02
        or abs(thumb_tip.y - thumb_ip.y) > 0.03
        or abs(thumb_tip.x - index_mcp.x) > 0.08
    )

    # Daca cele 4 degete principale sunt ridicate, acceptam palma chiar si
    # atunci cand degetul mare este partial ascuns de perspectiva camerei.
    return (raised_count == 4) or thumb_open


class SwipeDetector:
    def __init__(
        self,
        direction: int,
        min_delta_x: float,
        window_seconds: float,
        cooldown_seconds: float,
        min_samples: int,
    ) -> None:
        self._direction = 1 if direction >= 0 else -1
        self._min_delta_x = min_delta_x
        self._window_seconds = window_seconds
        self._cooldown_seconds = cooldown_seconds
        self._min_samples = min_samples
        self._history: Deque[Tuple[float, float]] = deque()
        self._last_trigger = 0.0

    def update(self, x_norm: float, now: float, pose_active: bool = True) -> bool:
        if not pose_active:
            self._history.clear()
            return False

        self._history.append((now, x_norm))

        while self._history and (now - self._history[0][0]) > self._window_seconds:
            self._history.popleft()

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if len(self._history) < self._min_samples:
            return False

        oldest_x = self._history[0][1]
        newest_x = self._history[-1][1]
        delta_x = (newest_x - oldest_x) * self._direction
        if delta_x < self._min_delta_x:
            return False

        # Filtreaza miscari oscilante: o parte mare din traseu nu trebuie
        # sa se intoarca in directia opusa swipe-ului dorit.
        backward_total = 0.0
        previous_x = self._history[0][1]
        for _, current_x in list(self._history)[1:]:
            step = (current_x - previous_x) * self._direction
            backward_total += max(0.0, -step)
            previous_x = current_x

        if backward_total > (self._min_delta_x * 0.35):
            return False

        self._last_trigger = now
        self._history.clear()
        return True


class OpenPalmDetector:
    def __init__(
        self,
        min_palm_span_x: float,
        hold_seconds: float,
        cooldown_seconds: float,
    ) -> None:
        self._min_palm_span_x = min_palm_span_x
        self._hold_seconds = hold_seconds
        self._cooldown_seconds = cooldown_seconds
        self._open_since: float | None = None
        self._last_trigger = 0.0

    def _is_pause_open_palm(self, hand_landmarks, mp_hands) -> bool:
        # Detector relaxat pentru comanda pause/play.
        lm = hand_landmarks.landmark
        index_up = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[
            mp_hands.HandLandmark.INDEX_FINGER_PIP
        ].y
        middle_up = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP
        ].y
        ring_up = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[
            mp_hands.HandLandmark.RING_FINGER_PIP
        ].y
        pinky_up = lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[
            mp_hands.HandLandmark.PINKY_PIP
        ].y
        raised_count = int(index_up) + int(middle_up) + int(ring_up) + int(pinky_up)
        if raised_count < 3:
            return False

        palm_span_x = abs(
            lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
            - lm[mp_hands.HandLandmark.PINKY_MCP].x
        )
        if palm_span_x < self._min_palm_span_x:
            return False

        thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
        index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        thumb_open = (
            abs(thumb_tip.x - thumb_ip.x) > 0.02
            or abs(thumb_tip.y - thumb_ip.y) > 0.03
            or abs(thumb_tip.x - index_mcp.x) > 0.08
        )
        # Accepta si cazurile in care degetul mare este partial ascuns.
        return thumb_open or raised_count == 4

    def update(self, hand_landmarks, mp_hands, now: float) -> bool:
        if not self._is_pause_open_palm(hand_landmarks, mp_hands):
            self._open_since = None
            return False

        if self._open_since is None:
            self._open_since = now
            return False

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if (now - self._open_since) < self._hold_seconds:
            return False

        self._last_trigger = now
        self._open_since = now
        return True


class TwoOpenPalmsHoldDetector:
    def __init__(self, hold_seconds: float, cooldown_seconds: float, min_palm_span_x: float) -> None:
        self._hold_seconds = hold_seconds
        self._cooldown_seconds = cooldown_seconds
        self._min_palm_span_x = min_palm_span_x
        self._open_since: float | None = None
        self._last_trigger = 0.0

    def update(self, all_hand_landmarks, mp_hands, now: float) -> bool:
        if not all_hand_landmarks or len(all_hand_landmarks) < 2:
            self._open_since = None
            return False

        first_open = _is_open_palm_pose(
            all_hand_landmarks[0], mp_hands, self._min_palm_span_x
        )
        second_open = _is_open_palm_pose(
            all_hand_landmarks[1], mp_hands, self._min_palm_span_x
        )
        if not (first_open and second_open):
            self._open_since = None
            return False

        if self._open_since is None:
            self._open_since = now
            return False

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if (now - self._open_since) < self._hold_seconds:
            return False

        self._last_trigger = now
        self._open_since = now
        return True


class IndexUpMotionDetector:
    def __init__(
        self,
        min_delta_y: float,
        window_seconds: float,
        cooldown_seconds: float,
        min_samples: int,
    ) -> None:
        self._min_delta_y = min_delta_y
        self._window_seconds = window_seconds
        self._cooldown_seconds = cooldown_seconds
        self._min_samples = min_samples
        self._history: Deque[Tuple[float, float]] = deque()
        self._last_trigger = 0.0

    def update(self, y_norm: float, pose_active: bool, now: float) -> bool:
        if not pose_active:
            self._history.clear()
            return False

        self._history.append((now, y_norm))
        while self._history and (now - self._history[0][0]) > self._window_seconds:
            self._history.popleft()

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if len(self._history) < self._min_samples:
            return False

        oldest_y = self._history[0][1]
        newest_y = self._history[-1][1]
        delta_up = oldest_y - newest_y
        if delta_up < self._min_delta_y:
            return False

        # Filtreaza oscilatiile puternice in directia opusa.
        backward_total = 0.0
        previous_y = self._history[0][1]
        for _, current_y in list(self._history)[1:]:
            backward_total += max(0.0, current_y - previous_y)
            previous_y = current_y
        if backward_total > (self._min_delta_y * 0.35):
            return False

        self._last_trigger = now
        self._history.clear()
        return True


class IndexDownHoldDetector:
    def __init__(self, hold_seconds: float, cooldown_seconds: float) -> None:
        self._hold_seconds = hold_seconds
        self._cooldown_seconds = cooldown_seconds
        self._down_since: float | None = None
        self._last_trigger = 0.0

    def update(self, pose_active: bool, now: float) -> bool:
        if not pose_active:
            self._down_since = None
            return False

        if self._down_since is None:
            self._down_since = now
            return False

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if (now - self._down_since) < self._hold_seconds:
            return False

        self._last_trigger = now
        self._down_since = now
        return True


class PoseHoldDetector:
    def __init__(self, hold_seconds: float, cooldown_seconds: float) -> None:
        self._hold_seconds = hold_seconds
        self._cooldown_seconds = cooldown_seconds
        self._active_since: float | None = None
        self._last_trigger = 0.0

    def update(self, pose_active: bool, now: float) -> bool:
        if not pose_active:
            self._active_since = None
            return False

        if self._active_since is None:
            self._active_since = now
            return False

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if (now - self._active_since) < self._hold_seconds:
            return False

        self._last_trigger = now
        self._active_since = now
        return True


class CrossedIndexFingersDetector:
    def __init__(self, hold_seconds: float, cooldown_seconds: float) -> None:
        self._hold_seconds = hold_seconds
        self._cooldown_seconds = cooldown_seconds
        self._cross_since: float | None = None
        self._last_trigger = 0.0

    def _is_index_up(self, hand_landmarks, mp_hands) -> bool:
        lm = hand_landmarks.landmark
        # Varianta mai toleranta la perspective diferite ale mainii.
        return lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[
            mp_hands.HandLandmark.INDEX_FINGER_PIP
        ].y

    def _orientation(self, a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _segments_intersect(self, a1, a2, b1, b2) -> bool:
        o1 = self._orientation(a1, a2, b1)
        o2 = self._orientation(a1, a2, b2)
        o3 = self._orientation(b1, b2, a1)
        o4 = self._orientation(b1, b2, a2)
        tolerance = 1e-4
        return (o1 * o2 <= tolerance) and (o3 * o4 <= tolerance)

    def _distance_point_to_segment(self, p, a, b) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        abx = bx - ax
        aby = by - ay
        ab_len_sq = (abx * abx) + (aby * aby)
        if ab_len_sq <= 1e-8:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = ax + (t * abx)
        proj_y = ay + (t * aby)
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    def _is_crossed_index_x(self, hand_a, hand_b, mp_hands) -> bool:
        if not self._is_index_up(hand_a, mp_hands) or not self._is_index_up(
            hand_b, mp_hands
        ):
            return False

        a_lm = hand_a.landmark
        b_lm = hand_b.landmark
        a_tip = (
            a_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            a_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
        )
        a_mcp = (
            a_lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            a_lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        )
        b_tip = (
            b_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
            b_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
        )
        b_mcp = (
            b_lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
            b_lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
        )
        fingertips_distance = abs(a_tip[0] - b_tip[0]) + abs(a_tip[1] - b_tip[1])
        if fingertips_distance > 0.55:
            return False

        if self._segments_intersect(a_mcp, a_tip, b_mcp, b_tip):
            return True

        # Fallback tolerant: varfurile sunt foarte aproape de segmentul opus.
        a_tip_to_b = self._distance_point_to_segment(a_tip, b_mcp, b_tip)
        b_tip_to_a = self._distance_point_to_segment(b_tip, a_mcp, a_tip)
        return (a_tip_to_b < 0.055) and (b_tip_to_a < 0.055)

    def update(self, all_hand_landmarks, mp_hands, now: float) -> bool:
        if not all_hand_landmarks or len(all_hand_landmarks) < 2:
            self._cross_since = None
            return False

        hand_a = all_hand_landmarks[0]
        hand_b = all_hand_landmarks[1]
        if not self._is_crossed_index_x(hand_a, hand_b, mp_hands):
            self._cross_since = None
            return False

        if self._cross_since is None:
            self._cross_since = now
            return False

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if (now - self._cross_since) < self._hold_seconds:
            return False

        self._last_trigger = now
        self._cross_since = now
        return True


class TimerCircleGestureDetector:
    def __init__(
        self,
        window_seconds: float,
        min_span: float,
        cooldown_seconds: float,
        min_samples: int,
    ) -> None:
        self._window_seconds = window_seconds
        self._min_span = min_span
        self._cooldown_seconds = cooldown_seconds
        self._min_samples = min_samples
        self._history: Deque[Tuple[float, float, float, bool]] = deque()
        self._last_trigger = 0.0

    def _is_open_palm(self, hand_landmarks, mp_hands) -> bool:
        lm = hand_landmarks.landmark
        index_up = (
            lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        )
        middle_up = (
            lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        )
        ring_up = (
            lm[mp_hands.HandLandmark.RING_FINGER_TIP].y
            < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
        )
        pinky_up = (
            lm[mp_hands.HandLandmark.PINKY_TIP].y
            < lm[mp_hands.HandLandmark.PINKY_PIP].y
        )
        thumb_open = (
            abs(
                lm[mp_hands.HandLandmark.THUMB_TIP].x
                - lm[mp_hands.HandLandmark.THUMB_IP].x
            )
            > 0.04
        )
        return index_up and middle_up and ring_up and pinky_up and thumb_open

    def update(self, hand_landmarks, mp_hands, now: float) -> bool:
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        is_open_palm = self._is_open_palm(hand_landmarks, mp_hands)
        self._history.append((now, wrist.x, wrist.y, is_open_palm))

        while self._history and (now - self._history[0][0]) > self._window_seconds:
            self._history.popleft()

        if (now - self._last_trigger) < self._cooldown_seconds:
            return False

        if len(self._history) < self._min_samples:
            return False

        open_count = sum(1 for _, _, _, open_palm in self._history if open_palm)
        if open_count < int(len(self._history) * 0.8):
            return False

        xs = [x for _, x, _, _ in self._history]
        ys = [y for _, _, y, _ in self._history]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        min_span = min(span_x, span_y)
        max_span = max(span_x, span_y)
        if min_span < self._min_span:
            return False

        if max_span > (min_span * 2.2):
            return False

        start_x = self._history[0][1]
        start_y = self._history[0][2]
        end_x = self._history[-1][1]
        end_y = self._history[-1][2]
        start_end_distance = abs(end_x - start_x) + abs(end_y - start_y)
        if start_end_distance > (min_span * 0.9):
            return False

        self._last_trigger = now
        self._history.clear()
        return True


def _is_index_only_up_pose(hand_landmarks, mp_hands) -> bool:
    lm = hand_landmarks.landmark
    index_up = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    )
    middle_down = (
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )
    ring_down = (
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_down = (
        lm[mp_hands.HandLandmark.PINKY_TIP].y
        > lm[mp_hands.HandLandmark.PINKY_PIP].y
    )
    return index_up and middle_down and ring_down and pinky_down


def _is_all_fingers_up_pose(hand_landmarks, mp_hands) -> bool:
    lm = hand_landmarks.landmark
    index_up = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    )
    middle_up = (
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )
    ring_up = (
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y
        < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_up = (
        lm[mp_hands.HandLandmark.PINKY_TIP].y
        < lm[mp_hands.HandLandmark.PINKY_PIP].y
    )
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_up = (
        thumb_tip.y < thumb_ip.y
        or abs(thumb_tip.x - index_mcp.x) > 0.08
    )
    return index_up and middle_up and ring_up and pinky_up and thumb_up


def _is_index_only_down_pose(hand_landmarks, mp_hands) -> bool:
    lm = hand_landmarks.landmark
    index_down = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        > lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    )
    middle_down = (
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )
    ring_down = (
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_down = (
        lm[mp_hands.HandLandmark.PINKY_TIP].y
        > lm[mp_hands.HandLandmark.PINKY_PIP].y
    )
    return index_down and middle_down and ring_down and pinky_down


def _draw_gesture_menu(frame) -> None:
    # Deseneaza in partea dreapta un meniu compact cu gesturile disponibile
    h, w, _ = frame.shape
    menu_width = 340
    padding = 14
    x0 = max(8, w - menu_width - 80)
    y0 = 10
    x1 = w - 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 22

    entries = [
        "Swipe dreapta (toate degetele) -> +5s YouTube",
        "Swipe stanga (toate degetele)  -> -5s YouTube",
        "Palma deschisa (o mana)      -> Play/Pause",
        "2 palme deschise             -> View/hide timer",
        "2 degete ridicate (cu timer)   -> +5 min timer",
        "Pumn inchis (cu timer)       -> Play/Pause timer",
        "Aratator in sus in miscare     -> Volum +",
        "Aratator in jos mentinut       -> Volum -",
        "2 aratatoare incrucisate (X)   -> Ctrl+W ",
        "TASTA Q - inchide aplicatia",
    ]
    # Calcul dinamic al inaltimii pentru a incadra toate liniile
    y1 = y0 + padding + 12 + (len(entries) + 1) * line_h + padding

    cv2.rectangle(frame, (x0, y0), (x1, y1), (15, 15, 15), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)

    x_text = x0 + padding
    y_text = y0 + padding + 4

    cv2.putText(frame, "Gesturi disponibile", (x_text, y_text), font, 0.55, (0, 215, 255), 2)
    y_text += line_h
    for idx, entry in enumerate(entries):
        is_last = idx == len(entries) - 1
        color = (0, 0, 255) if is_last else (255, 255, 255)
        scale = 0.52 if is_last else 0.48
        thickness = 2 if is_last else 1
        cv2.putText(
            frame,
            entry,
            (x_text, y_text),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y_text += line_h


def _is_two_fingers_up_pose(hand_landmarks, mp_hands) -> bool:
    lm = hand_landmarks.landmark
    index_up = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    )
    middle_up = (
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )
    ring_down = (
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_down = (
        lm[mp_hands.HandLandmark.PINKY_TIP].y
        > lm[mp_hands.HandLandmark.PINKY_PIP].y
    )
    return index_up and middle_up and ring_down and pinky_down


def _is_closed_fist_pose(hand_landmarks, mp_hands) -> bool:
    lm = hand_landmarks.landmark
    index_down = (
        lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    )
    middle_down = (
        lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )
    ring_down = (
        lm[mp_hands.HandLandmark.RING_FINGER_TIP].y
        > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_down = (
        lm[mp_hands.HandLandmark.PINKY_TIP].y
        > lm[mp_hands.HandLandmark.PINKY_PIP].y
    )

    # Degetul mare sta retras spre palma pentru pumn inchis.
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_to_index = abs(thumb_tip.x - index_mcp.x) + abs(thumb_tip.y - index_mcp.y)
    thumb_folded = thumb_to_index < 0.2
    return index_down and middle_down and ring_down and pinky_down and thumb_folded


def main() -> None:
    # Parametri de configurare pentru camera si detectie
    # Pot fi schimbati din fisierul .env
    camera_index = _read_int_env("CAMERA_INDEX", 0)
    frame_width = _read_int_env("FRAME_WIDTH", 1280)
    frame_height = _read_int_env("FRAME_HEIGHT", 720)
    display_window_width = _read_int_env("DISPLAY_WINDOW_WIDTH", 960)
    display_window_height = _read_int_env("DISPLAY_WINDOW_HEIGHT", 540)
    min_detection_confidence = _read_float_env("MIN_DETECTION_CONFIDENCE", 0.5)
    min_tracking_confidence = _read_float_env("MIN_TRACKING_CONFIDENCE", 0.5)
    swipe_right_min_delta_x = _read_float_env("SWIPE_RIGHT_MIN_DELTA_X", 0.18)
    swipe_right_window_seconds = _read_float_env("SWIPE_RIGHT_WINDOW_SECONDS", 0.45)
    swipe_right_cooldown_seconds = _read_float_env(
        "SWIPE_RIGHT_COOLDOWN_SECONDS", 1.0
    )
    swipe_right_min_samples = _read_int_env("SWIPE_RIGHT_MIN_SAMPLES", 5)
    open_palm_min_span_x = _read_float_env("OPEN_PALM_MIN_SPAN_X", 0.12)
    open_palm_hold_seconds = _read_float_env("OPEN_PALM_HOLD_SECONDS", 0.35)
    open_palm_cooldown_seconds = _read_float_env("OPEN_PALM_COOLDOWN_SECONDS", 1.2)
    pause_open_palm_min_span_x = _read_float_env("PAUSE_OPEN_PALM_MIN_SPAN_X", 0.12)
    timer_activate_two_palms_hold_seconds = _read_float_env(
        "TIMER_ACTIVATE_TWO_PALMS_HOLD_SECONDS", 0.35
    )
    timer_activate_two_palms_cooldown_seconds = _read_float_env(
        "TIMER_ACTIVATE_TWO_PALMS_COOLDOWN_SECONDS", 1.0
    )
    timer_add_five_hold_seconds = _read_float_env("TIMER_ADD_FIVE_HOLD_SECONDS", 0.3)
    timer_add_five_cooldown_seconds = _read_float_env(
        "TIMER_ADD_FIVE_COOLDOWN_SECONDS", 0.9
    )
    timer_start_fist_hold_seconds = _read_float_env(
        "TIMER_START_FIST_HOLD_SECONDS",
        _read_float_env("TIMER_START_THUMBS_HOLD_SECONDS", 0.28),
    )
    timer_start_fist_cooldown_seconds = _read_float_env(
        "TIMER_START_FIST_COOLDOWN_SECONDS",
        _read_float_env("TIMER_START_THUMBS_COOLDOWN_SECONDS", 1.0),
    )
    volume_up_min_delta_y = _read_float_env("VOLUME_UP_MIN_DELTA_Y", 0.14)
    volume_up_window_seconds = _read_float_env("VOLUME_UP_WINDOW_SECONDS", 0.45)
    volume_up_cooldown_seconds = _read_float_env("VOLUME_UP_COOLDOWN_SECONDS", 0.8)
    volume_up_min_samples = _read_int_env("VOLUME_UP_MIN_SAMPLES", 4)
    volume_down_hold_seconds = _read_float_env("VOLUME_DOWN_HOLD_SECONDS", 0.28)
    volume_down_cooldown_seconds = _read_float_env("VOLUME_DOWN_COOLDOWN_SECONDS", 0.8)
    close_x_hold_seconds = _read_float_env("CLOSE_X_HOLD_SECONDS", 0.25)
    close_x_cooldown_seconds = _read_float_env("CLOSE_X_COOLDOWN_SECONDS", 1.0)

    swipe_right_detector = SwipeDetector(
        direction=1,
        min_delta_x=swipe_right_min_delta_x,
        window_seconds=swipe_right_window_seconds,
        cooldown_seconds=swipe_right_cooldown_seconds,
        min_samples=swipe_right_min_samples,
    )
    swipe_left_detector = SwipeDetector(
        direction=-1,
        min_delta_x=swipe_right_min_delta_x,
        window_seconds=swipe_right_window_seconds,
        cooldown_seconds=swipe_right_cooldown_seconds,
        min_samples=swipe_right_min_samples,
    )
    open_palm_detector = OpenPalmDetector(
        min_palm_span_x=pause_open_palm_min_span_x,
        hold_seconds=open_palm_hold_seconds,
        cooldown_seconds=open_palm_cooldown_seconds,
    )
    timer_activate_detector = TwoOpenPalmsHoldDetector(
        hold_seconds=timer_activate_two_palms_hold_seconds,
        cooldown_seconds=timer_activate_two_palms_cooldown_seconds,
        min_palm_span_x=open_palm_min_span_x,
    )
    close_x_detector = CrossedIndexFingersDetector(
        hold_seconds=close_x_hold_seconds,
        cooldown_seconds=close_x_cooldown_seconds,
    )
    timer_add_five_detector = PoseHoldDetector(
        hold_seconds=timer_add_five_hold_seconds,
        cooldown_seconds=timer_add_five_cooldown_seconds,
    )
    timer_start_detector = PoseHoldDetector(
        hold_seconds=timer_start_fist_hold_seconds,
        cooldown_seconds=timer_start_fist_cooldown_seconds,
    )
    volume_up_detector = IndexUpMotionDetector(
        min_delta_y=volume_up_min_delta_y,
        window_seconds=volume_up_window_seconds,
        cooldown_seconds=volume_up_cooldown_seconds,
        min_samples=volume_up_min_samples,
    )
    volume_down_detector = IndexDownHoldDetector(
        hold_seconds=volume_down_hold_seconds,
        cooldown_seconds=volume_down_cooldown_seconds,
    )
    timer_visible = False
    timer_seconds = 0.0
    timer_running = False
    timer_last_update = time.monotonic()

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
    window_title = "INOC Hands-Free Cooking Assistant"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, display_window_width, display_window_height)

    # Configureaza detectorul de maini
    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        # Bucla principala: citeste cadrele video pana la iesire
        while True:
            loop_now = time.monotonic()
            timer_finished_now = False
            elapsed = loop_now - timer_last_update
            timer_last_update = loop_now
            if timer_running and timer_seconds > 0.0:
                timer_seconds = max(0.0, timer_seconds - elapsed)
                if timer_seconds <= 0.0:
                    timer_running = False
                    timer_finished_now = True

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
            status_text = ""
            if timer_finished_now:
                status_text = "timer reached 00:00:00"
            primary_hand_landmarks = None
            timer_toggled_now = False
            if timer_activate_detector.update(
                results.multi_hand_landmarks, mp_hands, loop_now
            ):
                timer_toggled_now = True
                if timer_visible:
                    timer_visible = False
                    timer_running = False
                    status_text = "two open palms detected -> timer hidden"
                else:
                    timer_visible = True
                    timer_seconds = 0
                    timer_running = False
                    status_text = "two open palms detected -> timer shown at 00:00:00"
            elif (
                (not timer_visible)
                and results.multi_hand_landmarks
                and len(results.multi_hand_landmarks) >= 2
            ):
                first_open = _is_open_palm_pose(
                    results.multi_hand_landmarks[0], mp_hands, open_palm_min_span_x
                )
                second_open = _is_open_palm_pose(
                    results.multi_hand_landmarks[1], mp_hands, open_palm_min_span_x
                )
                if first_open or second_open:
                    status_text = "hold 2 open palms to show timer"

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    if i == 0:
                        primary_hand_landmarks = hand_landmarks
                        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                        index_tip_y = hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ].y
                        index_only_up_pose = _is_index_only_up_pose(hand_landmarks, mp_hands)
                        all_fingers_up_pose = _is_all_fingers_up_pose(
                            hand_landmarks, mp_hands
                        )
                        index_only_down_pose = _is_index_only_down_pose(
                            hand_landmarks, mp_hands
                        )
                        two_fingers_up_pose = _is_two_fingers_up_pose(
                            hand_landmarks, mp_hands
                        )
                        closed_fist_pose = _is_closed_fist_pose(hand_landmarks, mp_hands)
                        now = loop_now
                        if swipe_right_detector.update(
                            wrist_x, now, pose_active=all_fingers_up_pose
                        ):
                            if _send_right_arrow():
                                status_text = "swipe right detected -> +5s"
                            else:
                                status_text = "swipe detected (unsupported OS)"
                        elif swipe_left_detector.update(
                            wrist_x, now, pose_active=all_fingers_up_pose
                        ):
                            if _send_left_arrow():
                                status_text = "swipe left detected -> -5s"
                            else:
                                status_text = "swipe detected (unsupported OS)"
                        elif volume_up_detector.update(index_tip_y, index_only_up_pose, now):
                            if _send_volume_up():
                                status_text = "index up motion detected -> volume up"
                            else:
                                status_text = "gesture detected (unsupported OS)"
                        elif volume_down_detector.update(index_only_down_pose, now):
                            if _send_volume_down():
                                status_text = "index down detected -> volume down"
                            else:
                                status_text = "gesture detected (unsupported OS)"
                        elif timer_visible and timer_add_five_detector.update(
                            two_fingers_up_pose, now
                        ):
                            timer_seconds += 5 * 60
                            status_text = "2 fingers up detected -> timer +5 min"
                        elif timer_visible and timer_start_detector.update(
                            closed_fist_pose, now
                        ):
                            if timer_seconds <= 0:
                                status_text = "closed fist detected -> set minutes first"
                            elif timer_running:
                                timer_running = False
                                status_text = "closed fist detected -> timer paused"
                            else:
                                timer_running = True
                                timer_last_update = now
                                status_text = "closed fist detected -> timer started"
                        elif (not timer_toggled_now) and open_palm_detector.update(
                            hand_landmarks, mp_hands, now
                        ):
                            if _send_k_key():
                                status_text = "open palm detected -> pause/play"
                            else:
                                status_text = "gesture detected (unsupported OS)"

            if close_x_detector.update(results.multi_hand_landmarks, mp_hands, loop_now):
                if _send_ctrl_w():
                    status_text = "index fingers X detected -> close YouTube tab"
                else:
                    status_text = "gesture detected (unsupported OS)"

            display_seconds = int(timer_seconds)
            timer_hours = display_seconds // 3600
            timer_minutes = (display_seconds % 3600) // 60
            timer_secs = display_seconds % 60
            timer_text = f"TIMER  {timer_hours:02d}:{timer_minutes:02d}:{timer_secs:02d}"

            # Afiseaza doar timerul in preview, fara alte texte.
            if timer_visible:
                cv2.rectangle(frame, (16, 16), (510, 68), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    timer_text,
                    (28, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
            # Meniu gesturi in partea dreapta
            _draw_gesture_menu(frame)
            # fereastra cu rezultatul procesarii
            cv2.imshow(window_title, frame)

            # Iesire din aplicatie la apasarea tastei Q.
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break

    # Elibereaza camera si inchide toate ferestrele OpenCV
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
