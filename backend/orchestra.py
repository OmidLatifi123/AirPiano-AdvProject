import os
import ctypes

# Add the path to the directory containing 'libfluidsynth-3.dll'
dll_path = r'C:\Tools\fluidsynth\fluidsynth-2.4.1-win10-x64\bin'
os.add_dll_directory(dll_path)

# Explicitly load the DLL
try:
    ctypes.cdll.LoadLibrary(os.path.join(dll_path, 'libfluidsynth-3.dll'))
    print("libfluidsynth-3.dll loaded successfully.")
except OSError as e:
    print(f"Error loading libfluidsynth-3.dll: {e}")
    exit(1)

# Import fluidsynth after ensuring the DLL is loaded
import fluidsynth
import cv2
import mediapipe as mp
import numpy as np


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize FluidSynth
sf2_path = "GeneralUser_GS.sf2"  # Path to your SoundFont file
synth = fluidsynth.Synth()
synth.start(driver="dsound")  # Change "dsound" to "alsa" on Linux
synth.sfload(sf2_path)
synth.program_select(0, 0, 52, 0)  # Set instrument to "Choir Ahhs" (52 in General MIDI)

# Configuration
MAX_PITCH = 72  # MIDI note for highest pitch (C5)
MIN_PITCH = 48  # MIDI note for lowest pitch (C3)
MAX_VOLUME = 127  # Maximum MIDI velocity
MIN_VOLUME = 30  # Minimum MIDI velocity
HARMONY_INTERVALS = [0, 4, 7]  # Root, major third, perfect fifth

# Track active notes for both hands
active_notes = {}

def calculate_pitch(y, frame_height):
    """Calculate root pitch based on hand height."""
    normalized_y = y / frame_height
    pitch = MIN_PITCH + int((1 - normalized_y) * (MAX_PITCH - MIN_PITCH))
    return pitch

def calculate_volume(z):
    """Calculate volume based on hand proximity (z-axis)."""
    volume = int(MAX_VOLUME - (z * (MAX_VOLUME - MIN_VOLUME)))
    return max(MIN_VOLUME, min(volume, MAX_VOLUME))

def play_chord(root_pitch, volume, hand_label):
    """Play a chord based on the root pitch and volume."""
    if hand_label not in active_notes:
        active_notes[hand_label] = set()

    new_notes = set(root_pitch + interval for interval in HARMONY_INTERVALS)
    current_notes = active_notes[hand_label]

    # Stop notes that are no longer part of the chord
    for note in current_notes - new_notes:
        stop_midi(note, hand_label)

    # Start notes that are new to the chord
    for note in new_notes - current_notes:
        play_midi(note, volume)

    # Update active notes
    active_notes[hand_label] = new_notes

def play_midi(note, volume):
    """Play a single MIDI note."""
    synth.noteon(0, note, volume)

def stop_midi(note, hand_label):
    """Stop a single MIDI note."""
    synth.noteoff(0, note)
    active_notes[hand_label].discard(note)

# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_class in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for the wrist
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Get coordinates and depth (z-axis) of the wrist
            wrist_y = int(wrist.y * frame_height)
            wrist_z = wrist.z

            # Determine pitch and volume based on hand position
            root_pitch = calculate_pitch(wrist_y, frame_height)
            volume = calculate_volume(wrist_z)

            # Determine which hand (left or right)
            handedness = hand_class.classification[0].label

            # Play a chord for this hand
            play_chord(root_pitch, volume, handedness)

            # Display hand information
            cv2.putText(frame, f"{handedness}: Root={root_pitch} Vol={volume}",
                        (10, 50 if handedness == "Right" else 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Virtual Choir', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
synth.delete()
cv2.destroyAllWindows()
