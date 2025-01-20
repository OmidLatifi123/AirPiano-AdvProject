import cv2
import mediapipe as mp
from pyo import *

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pyo Server
server = Server().boot()
server.start()

# Choir Formant Synthesis
# Base frequency (F0) for vocal pitch
base_freq = 220.0

# Formants for choir-like vowel sounds
formant1 = Sine(freq=base_freq * 0.75, mul=0.1)
formant2 = Sine(freq=base_freq * 1.5, mul=0.08)
formant3 = Sine(freq=base_freq * 3.0, mul=0.05)

# Combine formants
choir_sound = formant1 + formant2 + formant3

# Add amplitude modulation (simulates choir dynamics)
choir_modulation = Sine(freq=0.3, mul=0.2, add=0.8) * choir_sound

# Add spatial reverb for realism
choir_reverb = Freeverb(choir_modulation, size=0.8, damp=0.5, bal=0.7).out()

# Webcam Feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirrored effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process frame for hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    left_hand_y, right_hand_y = None, None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Identify whether hand is left or right
            handedness = result.multi_handedness
            for idx, hand in enumerate(handedness):
                label = hand.classification[0].label
                if label == 'Right':
                    right_hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h
                elif label == 'Left':
                    left_hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Adjust pitch and volume if both hands are visible
    if left_hand_y and right_hand_y:
        # Map left hand's Y-position to pitch
        pitch = base_freq + ((h - left_hand_y) / h) * 300  # Adjust pitch range as needed
        formant1.setFreq(pitch * 0.75)
        formant2.setFreq(pitch * 1.5)
        formant3.setFreq(pitch * 3.0)

        # Map right hand's Y-position to volume
        volume = max(0, min(1, 1 - right_hand_y / h))
        choir_sound.mul = volume
    else:
        choir_sound.mul = 0.0  # Mute sound if hands are not detected

    # Display the frame
    cv2.imshow("Hand Gesture Choir", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
server.stop()
