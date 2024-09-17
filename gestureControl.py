import cv2
import mediapipe as mp
import pyautogui
import threading

# Initialize MediaPipe Hand and Drawing utilities
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Start Video Capture
cap = cv2.VideoCapture(0)

# Limit frame rate to 30 FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Store the previous gesture to avoid redundant key presses
previous_gesture = None

# Frame processing interval (process every 2nd frame)
frame_count = 0
process_interval = 2

# Function to process gesture
def process_gesture(landmarks, frame_shape):
    global previous_gesture
    num_fingers = 0
    x, y = frame_shape[1], frame_shape[0]
    
    # Thumb
    if landmarks[4][0] < landmarks[3][0]:
        num_fingers += 1
    # Index Finger
    if landmarks[8][1] < landmarks[6][1]:
        num_fingers += 1
    # Middle Finger
    if landmarks[12][1] < landmarks[10][1]:
        num_fingers += 1
    # Ring Finger
    if landmarks[16][1] < landmarks[14][1]:
        num_fingers += 1
    # Pinky Finger
    if landmarks[20][1] < landmarks[18][1]:
        num_fingers += 1

    gesture = None
    if num_fingers == 1:
        gesture = "Up"
    elif num_fingers == 2:
        gesture = "Down"
    elif num_fingers == 3:
        gesture = "Right"
    elif num_fingers == 4:
        gesture = "Left"
    else:
        gesture = "Nothing"

    # Print the gesture to the console
    print(f"Detected gesture: {gesture}")

    # Only trigger pyautogui events if the gesture changes
    if gesture != previous_gesture:
        print(f"Gesture changed to: {gesture}")
        previous_gesture = gesture
        pyautogui.keyUp("up")
        pyautogui.keyUp("down")
        pyautogui.keyUp("right")
        pyautogui.keyUp("left")

        if gesture == "Up":
            pyautogui.keyDown("up")
        elif gesture == "Down":
            pyautogui.keyDown("down")
        elif gesture == "Right":
            pyautogui.keyDown("right")
        elif gesture == "Left":
            pyautogui.keyDown("left")

# Main loop
while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect and resize for performance
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))  # Resize for better performance
    
    # Convert BGR to RGB for Mediapipe
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    framergb.flags.writeable = False

    # Process only every 'process_interval' frame
    if frame_count % process_interval == 0:
        result = hands.process(framergb)
        
        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                # Draw hand landmarks
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in handslms.landmark]

                # Run gesture detection in a separate thread
                thread = threading.Thread(target=process_gesture, args=(landmarks, frame.shape))
                thread.start()

    # Show the frame
    cv2.imshow("GestureControl", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
