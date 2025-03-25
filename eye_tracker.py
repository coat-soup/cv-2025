import cv2
import mediapipe as mp
import time
import os
from pathlib import Path

# Toggle view mode
FULL_VIEW = True  # Set to False for minimal style and True for TPA/TNPA tracking

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Landmark indices
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_TOP_BOTTOM = [386, 374]

# Webcam & Video Writer
cap = cv2.VideoCapture(0)
w, h = 640, 480
cap.set(3, w)
cap.set(4, h)
downloads_path = str(Path.home() / "Downloads")
output_path = os.path.join(downloads_path, "eye_tracking_session.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

# Utilities
def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def vertical_distance(p1, p2):
    return abs(p2[1] - p1[1])

# Blink detection
BLINK_THRESHOLD = 5
blink_count = 0
blinking = False

# Attention tracking
attention_time = 0
distraction_time = 0
last_timestamp = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    timestamp = time.time()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    gaze_label = ""
    status = "Unknown"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]

            # SAFETY CHECK: Skip if any landmarks are missing
            required = LEFT_IRIS + RIGHT_IRIS + LEFT_EYE + RIGHT_EYE + LEFT_EYE_TOP_BOTTOM + RIGHT_EYE_TOP_BOTTOM
            try:
                if any(i >= len(face_landmarks.landmark) for i in required):
                    continue

                def get_coords(indices):
                    return [(int(face_landmarks.landmark[i].x * w),
                             int(face_landmarks.landmark[i].y * h)) for i in indices]

                left_iris = get_coords(LEFT_IRIS)
                right_iris = get_coords(RIGHT_IRIS)
                left_eye = get_coords(LEFT_EYE)
                right_eye = get_coords(RIGHT_EYE)
                left_eye_tb = get_coords(LEFT_EYE_TOP_BOTTOM)
                right_eye_tb = get_coords(RIGHT_EYE_TOP_BOTTOM)

                left_iris_center = midpoint(left_iris[0], left_iris[2])
                right_iris_center = midpoint(right_iris[0], right_iris[2])

                def estimate_horizontal_direction(eye, iris_center):
                    eye_left, eye_right = eye
                    eye_width = eye_right[0] - eye_left[0]
                    iris_offset = iris_center[0] - eye_left[0]
                    ratio = iris_offset / eye_width
                    if ratio < 0.35:
                        return "Left"
                    elif ratio > 0.65:
                        return "Right"
                    else:
                        return "Center"

                def is_looking_up(eye_tb, iris_center):
                    eye_top, eye_bottom = eye_tb
                    eye_height = eye_bottom[1] - eye_top[1]
                    iris_offset = iris_center[1] - eye_top[1]
                    ratio = iris_offset / eye_height
                    return ratio < 0.4

                left_dir = estimate_horizontal_direction(left_eye, left_iris_center)
                right_dir = estimate_horizontal_direction(right_eye, right_iris_center)
                left_up = is_looking_up(left_eye_tb, left_iris_center)
                right_up = is_looking_up(right_eye_tb, right_iris_center)

                if left_dir == right_dir:
                    if left_up and right_up:
                        if left_dir == "Left":
                            gaze_label = "Top Left"
                        elif left_dir == "Right":
                            gaze_label = "Top Right"
                        else:
                            gaze_label = "Center"
                    else:
                        gaze_label = left_dir
                else:
                    gaze_label = "Uncertain"

                # Blink detection
                left_eye_top, left_eye_bottom = left_eye_tb
                right_eye_top, right_eye_bottom = right_eye_tb
                left_dist = vertical_distance(left_eye_top, left_eye_bottom)
                right_dist = vertical_distance(right_eye_top, right_eye_bottom)

                if left_dist < BLINK_THRESHOLD and right_dist < BLINK_THRESHOLD:
                    if not blinking:
                        blink_count += 1
                        blinking = True
                else:
                    blinking = False

                # Attention status
                if gaze_label == "Center":
                    status = "Paying Attention"
                    label_color = (0, 255, 0)
                else:
                    status = "Not Paying Attention"
                    label_color = (0, 0, 255)

                # Time tracking
                if last_timestamp is not None:
                    frame_duration = timestamp - last_timestamp
                    if status == "Paying Attention":
                        attention_time += frame_duration
                    else:
                        distraction_time += frame_duration
                last_timestamp = timestamp

                # Draw iris centers
                cv2.circle(frame, left_iris_center, 3, label_color, -1)
                cv2.circle(frame, right_iris_center, 3, label_color, -1)

                # Display overlays
                cv2.putText(frame, f"Looking: {gaze_label}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

                if blinking:
                    cv2.putText(frame, "Blinking", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if FULL_VIEW:
                    tpa_tnpa = f"TPA: {int(attention_time)}s   TNPA: {int(distraction_time)}s"
                    cv2.putText(frame, tpa_tnpa,
                                (frame.shape[1] - 280, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                print(f"Skipped frame: {e}")
                continue

    out.write(frame)
    cv2.imshow("Gaze & Blink Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f" Video saved to: {output_path}")
