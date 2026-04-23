import cv2
import numpy as np
from collections import deque

CAMERA_INDEX = 0
MIN_MOVEMENT_PIXELS = 35
SMOOTH_WINDOW = 5
SHOW_TRAIL = True


def create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    raise AttributeError("No supported OpenCV tracker found.")


class RepCounter:
    def __init__(self, min_movement_pixels=35, smooth_window=5):
        self.reps = 0
        self.state = "DOWN"
        self.min_movement_pixels = min_movement_pixels
        self.y_history = deque(maxlen=smooth_window)
        self.trail = deque(maxlen=40)

        self.start_y = None
        self.lowest_y = None
        self.highest_y = None
        self.last_smooth_y = None

    def reset(self):
        self.reps = 0
        self.state = "DOWN"
        self.y_history.clear()
        self.trail.clear()
        self.start_y = None
        self.lowest_y = None
        self.highest_y = None
        self.last_smooth_y = None

    def update(self, center_y):
        self.y_history.append(center_y)
        self.trail.append(center_y)

        if len(self.y_history) < self.y_history.maxlen:
            return self.reps, self.state, None, 0, 0

        smooth_y = int(np.mean(self.y_history))

        if self.start_y is None:
            self.start_y = smooth_y
            self.lowest_y = smooth_y
            self.highest_y = smooth_y
            self.last_smooth_y = smooth_y
            return self.reps, self.state, smooth_y, 0, 0

        self.lowest_y = min(self.lowest_y, smooth_y)
        self.highest_y = max(self.highest_y, smooth_y)

        up_distance = self.highest_y - smooth_y
        down_distance = smooth_y - self.lowest_y

        if self.state == "DOWN":
            if up_distance > self.min_movement_pixels:
                self.state = "UP"

        elif self.state == "UP":
            if down_distance > self.min_movement_pixels:
                self.reps += 1
                self.state = "DOWN"
                self.lowest_y = smooth_y
                self.highest_y = smooth_y

        self.last_smooth_y = smooth_y
        return self.reps, self.state, smooth_y, up_distance, down_distance


def draw_text(frame, lines, x=20, y=35, line_gap=32, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    tracker = None
    bbox = None
    rep_counter = RepCounter(
        min_movement_pixels=MIN_MOVEMENT_PIXELS,
        smooth_window=SMOOTH_WINDOW
    )

    print("Instructions:")
    print("1. Press R to select the moving body part")
    print("2. Draw a box around your hand, forearm, dumbbell, or torso")
    print("3. Press ENTER or SPACE")
    print("4. Press C to reset reps")
    print("5. Press Q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if tracker is None:
            draw_text(display, [
                "Mode: WAITING",
                "Press R to select moving area",
                "Press Q to quit"
            ])
        else:
            ok, bbox = tracker.update(frame)

            if ok:
                x, y, w, h = [int(v) for v in bbox]
                cx = x + w // 2
                cy = y + h // 2

                reps, state, smooth_y, up_distance, down_distance = rep_counter.update(cy)

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)

                if SHOW_TRAIL and len(rep_counter.trail) > 1:
                    trail_list = list(rep_counter.trail)
                    for i in range(1, len(trail_list)):
                        p1 = (cx, trail_list[i - 1])
                        p2 = (cx, trail_list[i])
                        cv2.line(display, p1, p2, (255, 255, 0), 2)

                draw_text(display, [
                    "Mode: TRACKING",
                    f"Reps: {reps}",
                    f"State: {state}",
                    f"Y: {smooth_y if smooth_y is not None else '-'}",
                    f"Up movement: {up_distance}",
                    f"Down movement: {down_distance}",
                    "Press R to reselect ROI",
                    "Press C to reset reps",
                    "Press Q to quit"
                ], color=(255, 255, 255))

            else:
                draw_text(display, [
                    "Tracking lost",
                    "Press R to reselect ROI",
                    "Press Q to quit"
                ], color=(0, 0, 255))

        cv2.imshow("Exercise Rep Counter", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            rep_counter.reset()
            print("Rep counter reset.")

        elif key == ord("r"):
            paused_frame = frame.copy()
            bbox = cv2.selectROI("Exercise Rep Counter", paused_frame, False, False)
            if bbox != (0, 0, 0, 0):
                tracker = create_tracker()
                tracker.init(frame, bbox)
                rep_counter.reset()
                print("ROI selected and tracker initialized.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
