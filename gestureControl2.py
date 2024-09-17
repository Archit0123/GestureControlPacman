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

# Function to detect thumb direction and map to controls
def detect_thumb_direction(landmarks, frame_shape):
    global previous_gesture

    x, y = frame_shape[1], frame_shape[0]
    
    # Thumb IP joint (landmark 3) and Thumb tip (landmark 4)
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    gesture = None

    # Thumb Up (Y-coordinates: thumb tip above thumb IP)
    if thumb_tip[1] < thumb_ip[1]:
        gesture = "Up"
    
    # Thumb Down (Y-coordinates: thumb tip below thumb IP)
    elif thumb_tip[1] > thumb_ip[1]:
        gesture = "Down"
    
    # Thumb Right (X-coordinates: thumb tip right of thumb IP)
    elif (thumb_tip[1] > thumb_ip[1]) and (landmarks[8][1] < landmarks[6][1]):
        gesture = "Right"
    
    # Thumb Left (X-coordinates: thumb tip left of thumb IP)
    elif thumb_tip[0] < thumb_ip[0]:
        gesture = "Left"
    
    else:
        gesture = "Nothing"

    # Print the gesture to the console for debugging
    print(f"Detected thumb direction: {gesture}")

    # Trigger the gesture if it changes
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

                # Detect thumb direction and map to control
                thread = threading.Thread(target=detect_thumb_direction, args=(landmarks, frame.shape))
                thread.start()

    # Show the frame
    cv2.imshow("GestureControl", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
