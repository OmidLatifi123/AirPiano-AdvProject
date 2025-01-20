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

# Choir Ensemble: Multiple slightly detuned oscillators
base_freq = 220.0  # Base frequency for the choir (A3)

# Create a group of oscillators for ensemble effect
choir_oscillators = [
    Sine(freq=base_freq * (1 + i * 0.01), mul=0.1) for i in range(-3, 4)  # Slight detuning
]
choir_ensemble = Mix(choir_oscillators, voices=7)

# Add formant filters to simulate vowel sounds
formant1 = Biquad(choir_ensemble, freq=700, q=5)  # F1 for "ah"
formant2 = Biquad(formant1, freq=1200, q=3)  # F2 for "ah"

# Add amplitude modulation for vibrato
vibrato = Sine(freq=6, mul=0.01, add=1)  # Slow vibrato
choir_vibrato = formant2 * vibrato

# Add reverb for spatial realism
choir_reverb = Freeverb(choir_vibrato, size=0.9, damp=0.4, bal=0.7).out()

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
        for osc in choir_oscillators:
            osc.setFreq(pitch * (1 + osc.freq / base_freq * 0.02))  # Slight detuning

        # Map right hand's Y-position to volume
        volume = max(0, min(1, 1 - right_hand_y / h))
        choir_ensemble.mul = volume
    else:
        choir_ensemble.mul = 0.0  # Mute sound if hands are not detected

    # Display the frame
    cv2.imshow("Hand Gesture Choir", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
server.stop()
