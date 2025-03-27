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

# Smoothin params
move_speed = 0.1
screen_gaze = (0,0)

# Eye and iris landmark indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # Simplified for left eye
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]  # Simplified for right eye
LEFT_IRIS_IDX = [469, 470, 471, 472]  # Iris landmarks for left eye
RIGHT_IRIS_IDX = [474, 475, 476, 477]  # Iris landmarks for right eye
LEFT_EYE_TOP_BOTTOM_IDX = [386, 374]  # Top and bottom of the left eye
RIGHT_EYE_TOP_BOTTOM_IDX = [159, 145]  # Top and bottom of the right eye

def lerp(a,b,t):
    return a + (b-a) * t

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
    
    """
    # Apply a non-linear scaling to the vertical movement (like sensitivity)
    if iris_y_normalized < 0.5:
        # If iris is in the upper half, scale it upward more rapidly
        iris_y_normalized = 0.5 + (iris_y_normalized - 0.5) ** (1 / vertical_sensitivity)
        print("smal",iris_y_normalized)
    else:
        # If iris is in the lower half, scale it downward more rapidly
        iris_y_normalized = 0.5 - math.abs(iris_y_normalized - 0.5) ** (1 / vertical_sensitivity)
        print("big", iris_y_normalized)"""
    

    #iris_y_normalized = vs * (iris_y_normalized - 0.5) + 0.5

    iris_y_normalized = max(min(iris_y_normalized, 1), 0)

    bias, scale = -2000, 4

    # Map to screen coordinates
    screen_x = int(screen_width - iris_x_normalized * screen_width)
    screen_y = int(screen_height - iris_y_normalized * screen_height) * scale + bias
    
    return screen_x, screen_y

# Set window properties to fullscreen
cv2.namedWindow('Eye Tracking', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Eye Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
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

    # Resize the camera feed to 1/4th of the screen size
    small_frame = cv2.resize(frame, (screen_width // 4, screen_height // 4))
    
    # Place the resized camera feed in the top-left corner of the fullscreen frame
    fullscreen_frame[0:small_frame.shape[0], 0:small_frame.shape[1]] = small_frame

    # Display the fullscreen frame
    cv2.imshow('Eye Tracking', fullscreen_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
