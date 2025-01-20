import cv2
import mediapipe as mp
import pygame
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Pygame for Sound Playback
pygame.init()
pygame.mixer.init()
if not pygame.mixer.get_init():
    raise RuntimeError("Pygame mixer failed to initialize. Check your sound setup.")

# Load Choir Sound Files
choir_file = "sounds/choir1.mp3"
if not os.path.exists(choir_file):
    raise FileNotFoundError(f"Choir sound file not found: {choir_file}")

choir_left = pygame.mixer.Sound(choir_file)
choir_right = pygame.mixer.Sound(choir_file)

# Start capturing video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Smooth transition parameters
smooth_factor = 0.1
fade_speed = 0.05
pitch_scaling_factor = 4.0

# State variables
left_hand_visible = False
right_hand_visible = False
left_hand_fist = False
right_hand_fist = False
current_left_volume = 0.0
current_right_volume = 0.0
left_channel = pygame.mixer.Channel(0)
right_channel = pygame.mixer.Channel(1)

print("Camera initialized. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    left_hand_visible = False
    right_hand_visible = False
    left_hand_fist = False
    right_hand_fist = False
    target_left_volume = 0.0
    target_right_volume = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            y_position = index_tip.y * frame.shape[0]
            x_position = index_tip.x * frame.shape[1]

            center_x = frame.shape[1] / 2
            volume = max(0.1, 1.0 - abs(x_position - center_x) / center_x)
            pitch = max(0.5, pitch_scaling_factor - (y_position / frame.shape[0]) * pitch_scaling_factor)

            is_fist = all(
                hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 2].y
                for i in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                          mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                          mp_hands.HandLandmark.RING_FINGER_TIP,
                          mp_hands.HandLandmark.PINKY_TIP]
            )

            if is_fist:
                if hand_label == "Left":
                    left_hand_fist = True
                elif hand_label == "Right":
                    right_hand_fist = True

            if hand_label == "Left":
                left_hand_visible = True
                target_left_volume = volume
            elif hand_label == "Right":
                right_hand_visible = True
                target_right_volume = volume

    current_left_volume += smooth_factor * (target_left_volume - current_left_volume)
    current_right_volume += smooth_factor * (target_right_volume - current_right_volume)

    if not left_hand_fist:
        left_channel.set_volume(current_left_volume if left_hand_visible else 0.0)
    else:
        left_channel.fadeout(500)

    if not right_hand_fist:
        right_channel.set_volume(current_right_volume if right_hand_visible else 0.0)
    else:
        right_channel.fadeout(500)

    if left_hand_visible and not left_channel.get_busy():
        left_channel.play(choir_left, loops=-1)
    elif not left_hand_visible:
        left_channel.stop()

    if right_hand_visible and not right_channel.get_busy():
        right_channel.play(choir_right, loops=-1)
    elif not right_hand_visible:
        right_channel.stop()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Controllable Choir', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
left_channel.stop()
right_channel.stop()
pygame.quit()
cv2.destroyAllWindows()
