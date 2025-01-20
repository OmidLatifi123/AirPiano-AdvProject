import cv2
import mediapipe as mp
import numpy as np

from pyo import *

# ====== Mediapipe Setup ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# ====== Pyo Server Setup ======
s = Server().boot()
s.start()

# -------------------------------------------------------
#       VIOLIN STRING CLASS
# -------------------------------------------------------
class ViolinString:
    """
    A violin-like string sound using:
    - Sawtooth waveform (Blit or Saw)
    - ADSR envelope for bowing effect
    - Band-pass filter to shape the timbre
    """
    def __init__(self, base_freq, mul=0.2):
        self.base_freq = base_freq
        self.mul = mul

        # Oscillator for the base sound
        self.osc = Blit(freq=self.base_freq, harms=64, mul=self.mul)

        # Band-pass filter to shape the timbre
        self.filter = Biquad(self.osc, freq=self.base_freq, q=5, type=2)

        # ADSR envelope to simulate bowing
        self.env = Adsr(attack=0.1, decay=0.2, sustain=0.7, release=0.5, mul=self.mul)

        # Final sound output
        self.output = self.filter * self.env
        self.output.out()

    def set_pitch(self, freq):
        """
        Update the frequency of the oscillator and filter.
        """
        if freq > 0:
            self.osc.freq = freq
            self.filter.freq = freq

    def start(self):
        """
        Start the envelope (simulate bowing).
        """
        self.env.play()

    def stop(self):
        """
        Stop the envelope (simulate stopping the bow).
        """
        self.env.stop()

# -------------------------------------------------------
#       VIOLIN INSTRUMENT CLASS
# -------------------------------------------------------
class ViolinInstrument:
    """
    Manages 4 strings (G, D, A, E) for a violin:
      - G = 196 Hz
      - D = 293.66 Hz
      - A = 440 Hz
      - E = 659.25 Hz

    The user moves their hand across the screen:
      X => selects which string (4 vertical zones)
      Y => controls pitch offset from the base frequency (like finger placement).
    """
    def __init__(self):
        # Standard violin tuning (approx.)
        self.string_freqs = [196.0, 293.66, 440.0, 659.25]  # G, D, A, E
        self.string_labels = ["G", "D", "A", "E"]

        # Create 4 strings
        self.strings = [
            ViolinString(freq) for freq in self.string_freqs
        ]

        # Map vertical position to up to +12 semitones above base freq
        self.max_semitones_up = 12

        # Which string is currently active (None if no hand detected)
        self.active_string_idx = None

    def set_active_string(self, idx, pitch_factor):
        """
        Select the active string and set its pitch offset (in semitones).
        pitch_factor from 0..1 => 0 means no offset, 1 means +12 semitones.
        """
        if idx is None or idx < 0 or idx >= len(self.strings):
            return

        # Current string's base frequency
        base_freq = self.string_freqs[idx]

        # Map pitch_factor [0..1] => up to +12 semitones
        semitone_offset = pitch_factor * self.max_semitones_up
        freq_multiplier = 2 ** (semitone_offset / 12.0)
        new_freq = base_freq * freq_multiplier

        # Update the string pitch and start the sound
        self.strings[idx].set_pitch(new_freq)
        self.strings[idx].start()

        # Stop other strings
        for i, string in enumerate(self.strings):
            if i != idx:
                string.stop()

    def stop_all(self):
        """
        Stop generating sound from all strings.
        """
        for string in self.strings:
            string.stop()

# -------------------------------------------------------
#       MAIN
# -------------------------------------------------------
def main():
    print("Creating violin instrument...")
    violin = ViolinInstrument()

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    print("Starting main loop...")
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            # Draw 4 vertical "strings"
            num_strings = 4
            for i in range(num_strings):
                x_pos = int((i + 0.5) * (w / num_strings))
                color = (255, 255, 255)

                cv2.line(frame, (x_pos, 0), (x_pos, h), color, 2)
                cv2.putText(
                    frame,
                    violin.string_labels[i],
                    (x_pos - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

            # Track the right hand index fingertip
            right_hand_x = None
            right_hand_y = None

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                        result.multi_hand_landmarks,
                        result.multi_handedness):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    if handedness.classification[0].label == 'Right':
                        index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        right_hand_x = int(index_fingertip.x * w)
                        right_hand_y = int(index_fingertip.y * h)

                        # Draw a small circle for clarity
                        cv2.circle(
                            frame, (right_hand_x, right_hand_y),
                            10, (0, 255, 0), -1
                        )

            if right_hand_x is not None:
                # Determine active string based on X
                zone_width = w / num_strings
                string_index = int(right_hand_x // zone_width)

                # Map Y position to pitch factor
                if right_hand_y is not None:
                    pitch_factor = min(max(right_hand_y / h, 0.0), 1.0)
                    violin.set_active_string(string_index, pitch_factor)
            else:
                # No hand detected => stop all sound
                violin.stop_all()

            cv2.imshow("Violin Gestures", frame)

            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    print("Cleaning up...")
    violin.stop_all()
    cap.release()
    cv2.destroyAllWindows()
    s.stop()
    print("Program ended")

# Entry point
if __name__ == "__main__":
    main()
