import cv2
import mediapipe as mp
import pygame.midi
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize pygame MIDI
pygame.midi.init()
midi_out = pygame.midi.Output(0)

# Define MIDI pitch range and volume range
MIDI_PITCH_MIN = 40  # Low pitch (E2)
MIDI_PITCH_MAX = 80  # High pitch (G5)
MIDI_VOLUME_MIN = 30  # Soft volume
MIDI_VOLUME_MAX = 127  # Loud volume

# Track active notes
current_pitch = None
current_volume = None

# Start capturing video
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    pygame.midi.quit()
    exit()

print("Camera initialized. Press 'q' to exit.")

# Smooth transition parameters
smooth_factor = 0.1  # Smaller values make the transitions smoother

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally for a mirrored effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Reset pitch and volume targets
    target_pitch, target_volume = None, None

    # Detect hand landmarks and draw them
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label  # "Left" or "Right"

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index fingertip position
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            y_position = index_tip.y * frame.shape[0]

            # Map y-position to MIDI pitch and volume range
            pitch = int(MIDI_PITCH_MIN + (MIDI_PITCH_MAX - MIDI_PITCH_MIN) * (1 - y_position / frame.shape[0]))
            volume = int(MIDI_VOLUME_MIN + (MIDI_VOLUME_MAX - MIDI_VOLUME_MIN) * (1 - y_position / frame.shape[0]))

            # Set pitch for left hand and volume for right hand
            if hand_label == "Left":
                target_pitch = pitch
            elif hand_label == "Right":
                target_volume = volume

    # Smooth transitions for pitch and volume
    if target_pitch is not None:
        if current_pitch is None:
            current_pitch = target_pitch
        else:
            current_pitch += smooth_factor * (target_pitch - current_pitch)
        midi_out.note_on(int(current_pitch), velocity=100)

    if target_volume is not None:
        if current_volume is None:
            current_volume = target_volume
        else:
            current_volume += smooth_factor * (target_volume - current_volume)
        midi_out.write_short(0xB0, 7, int(current_volume))

    # Show the frame with hand landmarks
    cv2.imshow('Violin Hand Tracker', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
midi_out.note_off(int(current_pitch), velocity=100)
pygame.midi.quit()
cv2.destroyAllWindows()
