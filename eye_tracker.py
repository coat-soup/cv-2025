import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)


# Screen dimensions (adjust these for your display resolution)
screen_width = 1920
screen_height = 1080

# Heatmap
heatmap = np.zeros((screen_height, screen_width), dtype=np.float32)
time_running = 0

# Smoothin params
move_speed = 0.05
screen_gaze = (0,0)

# Eye and iris landmark indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_IDX = [469, 470, 471, 472]
RIGHT_IRIS_IDX = [474, 475, 476, 477]
LEFT_EYE_TOP_BOTTOM_IDX = [386, 374]
RIGHT_EYE_TOP_BOTTOM_IDX = [159, 145]

# Standard linear interpolation
def lerp(a,b,t):
    return a + (b-a) * t

# Returns limits if value outside of range
def clamp(val, _min, _max):
    return max(min(val, _max), _min)

# Function to calculate the center of the iris
def calculate_iris_center(iris_landmarks):
    x_coords = [landmark[0] for landmark in iris_landmarks]
    y_coords = [landmark[1] for landmark in iris_landmarks]
    return int(np.mean(x_coords)), int(np.mean(y_coords))


# Function to map gaze to screen coordinates
def map_gaze_to_screen(iris_center, eye_box, eye_top_bottom_dist, screen_width, screen_height, vs=5):
    eye_width = eye_box[1][0] - eye_box[0][0]
    
    # Normalize the iris position within the eye box (0,0) to (1,1)
    iris_x_normalized = (iris_center[0] - eye_box[0][0]) / eye_width
    iris_y_normalized = (iris_center[1] - eye_box[0][1]) / eye_top_bottom_dist    

    #iris_y_normalized = vs * (iris_y_normalized - 0.5) + 0.5

    iris_y_normalized = max(min(iris_y_normalized, 1), 0)

    bias, scale = -2500, 4

    # Map to screen coordinates
    screen_x = int(screen_width - iris_x_normalized * screen_width)
    screen_y = int(screen_height - iris_y_normalized * screen_height) * scale + bias

    #screen_x = clamp(screen_x, 0, screen_width)
    #screen_y = clamp(screen_y, 0, screen_height)
    print(screen_x, screen_y)

    return screen_x, screen_y

# Set window properties to fullscreen
cv2.namedWindow('Eye Tracking', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Eye Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    time_running += 0.1

    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to get face landmarks
    results = face_mesh.process(rgb_frame)
    
    # Create a black fullscreen frame
    fullscreen_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for the left and right eye
            left_eye = np.array([[int(face_landmarks.landmark[idx].x * frame.shape[1]), 
                                  int(face_landmarks.landmark[idx].y * frame.shape[0])] for idx in LEFT_EYE_IDX])
            right_eye = np.array([[int(face_landmarks.landmark[idx].x * frame.shape[1]), 
                                   int(face_landmarks.landmark[idx].y * frame.shape[0])] for idx in RIGHT_EYE_IDX])
            
            # Get iris landmarks
            left_iris = np.array([[int(face_landmarks.landmark[idx].x * frame.shape[1]), 
                                   int(face_landmarks.landmark[idx].y * frame.shape[0])] for idx in LEFT_IRIS_IDX])
            right_iris = np.array([[int(face_landmarks.landmark[idx].x * frame.shape[1]), 
                                    int(face_landmarks.landmark[idx].y * frame.shape[0])] for idx in RIGHT_IRIS_IDX])
            
            # Calculate iris centers
            left_iris_center = calculate_iris_center(left_iris)
            right_iris_center = calculate_iris_center(right_iris)

            # Get the top and bottom of the eye (for more accurate vertical movement)
            left_eye_top_bottom = np.array([[int(face_landmarks.landmark[idx].x * frame.shape[1]),
                                             int(face_landmarks.landmark[idx].y * frame.shape[0])] for idx in LEFT_EYE_TOP_BOTTOM_IDX])
            right_eye_top_bottom = np.array([[int(face_landmarks.landmark[idx].x * frame.shape[1]),
                                              int(face_landmarks.landmark[idx].y * frame.shape[0])] for idx in RIGHT_EYE_TOP_BOTTOM_IDX])

            # Calculate the distance between the top and bottom of the eyes
            left_eye_top_bottom_dist = np.linalg.norm(left_eye_top_bottom[0] - left_eye_top_bottom[1])
            right_eye_top_bottom_dist = np.linalg.norm(right_eye_top_bottom[0] - right_eye_top_bottom[1])
            
            # Map gaze to screen coordinates using the eye top/bottom distance for normalization
            left_eye_box = (np.min(left_eye, axis=0), np.max(left_eye, axis=0))
            right_eye_box = (np.min(right_eye, axis=0), np.max(right_eye, axis=0))
            
            left_screen_gaze = map_gaze_to_screen(left_iris_center, left_eye_box, left_eye_top_bottom_dist, screen_width, screen_height)
            right_screen_gaze = map_gaze_to_screen(right_iris_center, right_eye_box, right_eye_top_bottom_dist, screen_width, screen_height)
            
            # Calculate average gaze point on the screen
            avg_screen_gaze = (
                int((left_screen_gaze[0] + right_screen_gaze[0]) / 2),
                int((left_screen_gaze[1] + right_screen_gaze[1]) / 2)
            )
            screen_gaze = (
                int(lerp(screen_gaze[0], avg_screen_gaze[0], move_speed)),
                int(lerp(screen_gaze[1], avg_screen_gaze[1], move_speed))
            )
            
            # Draw the gaze point on the fullscreen frame
            cv2.circle(fullscreen_frame, screen_gaze, 10, (255, 0, 0), -1)

            # Draw to heatmap
            temp_heatmap = np.zeros_like(heatmap)
            cv2.circle(temp_heatmap, screen_gaze, 50, 1.0, -1)
            temp_heatmap = cv2.GaussianBlur(temp_heatmap, (101, 101), 0) # Blur for smoothing
            heatmap = cv2.add(heatmap, temp_heatmap)  # Add heatmap values (preserves previous values)

    
    # Normalize heatmap to 0-255 and apply a color map
    heatmap_norm = np.clip((heatmap/time_running) * 255.0, 0, 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Blend the colored heatmap with the fullscreen frame
    fullscreen_frame = cv2.addWeighted(colored_heatmap, 0.7, fullscreen_frame, 0.3, 0)
    
    # Resize the camera feed and place it in the corner
    small_frame = cv2.resize(frame, (screen_width // 4, screen_height // 4))
    fullscreen_frame[0:small_frame.shape[0], 0:small_frame.shape[1]] = small_frame

    # Display the fullscreen frame
    cv2.imshow('Eye Tracking', fullscreen_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
