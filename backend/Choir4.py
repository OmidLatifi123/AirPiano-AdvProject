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

# ====== ChoirVoice Class ======
class ChoirVoice:
    """
    A synthetic choir voice with three vowels:
      - "OO" for low pitches (< ~170 Hz)
      - "OH" for mid-low pitches (170–300 Hz)
      - "AH" for mid-high to high pitches (> 300 Hz)

    This helps mid-lower pitches sound more “human” than a single vowel approach.
    """
    def __init__(self, base_freq=220.0):
        self.base_freq = base_freq

        # --- 1) Base Oscillators (BLIT + some sine harmonics + pink noise) ---
        self.fundamental = Blit(freq=base_freq, harms=64, mul=0.3)
        self.fundamental_detuned1 = Blit(freq=base_freq * 0.99, harms=64, mul=0.2)
        self.fundamental_detuned2 = Blit(freq=base_freq * 1.01, harms=64, mul=0.2)

        # Mild harmonics (sine waves)
        self.harmonic1 = Sine(freq=base_freq * 2, mul=0.15)
        self.harmonic2 = Sine(freq=base_freq * 3, mul=0.1)
        self.harmonic3 = Sine(freq=base_freq * 4, mul=0.05)

        # Slight vibrato
        self.vibrato = Sine(freq=5, mul=3)
        self.vibrato_osc = Sine(freq=base_freq + self.vibrato, mul=0.2)

        # Gentle pink noise for breathiness
        self.noise = PinkNoise(mul=0.01)

        # Sum them up into a stereo signal
        self.raw_mix = Mix(
            [
                self.fundamental,
                self.fundamental_detuned1,
                self.fundamental_detuned2,
                self.harmonic1,
                self.harmonic2,
                self.harmonic3,
                self.vibrato_osc,
                self.noise
            ],
            voices=2
        )

        # --- 2) Dynamic Formants for "OO," "OH," and "AH" ---
        # We'll define three sets of formants:
        #   "OO" -> F1=350, F2=600,  F3=2400, F4=3000
        #   "OH" -> F1=450, F2=850,  F3=2400, F4=2800
        #   "AH" -> F1=600, F2=1000, F3=2400, F4=2800
        #
        # We'll select which set to use based on current pitch.

        # We use SigTo for smooth transitions of each formant freq.
        self.formant_f1 = SigTo(value=350, time=0.2, init=350)
        self.formant_f2 = SigTo(value=600, time=0.2, init=600)
        self.formant_f3 = SigTo(value=2400, time=0.2, init=2400)
        self.formant_f4 = SigTo(value=3000, time=0.2, init=3000)

        # Lower Q to reduce squeakiness or metallic timbre
        self.bpf1 = Biquad(self.raw_mix, freq=self.formant_f1, q=8, type=2)
        self.bpf2 = Biquad(self.raw_mix, freq=self.formant_f2, q=8, type=2)
        self.bpf3 = Biquad(self.raw_mix, freq=self.formant_f3, q=8, type=2)
        self.bpf4 = Biquad(self.raw_mix, freq=self.formant_f4, q=8, type=2)

        # Parallel mix of all 4 band-pass filters
        self.formants_sum = Mix(
            [self.bpf1, self.bpf2, self.bpf3, self.bpf4],
            voices=2
        )

        # Combine some raw signal for brightness
        self.colored_mix = Mix([self.raw_mix * 0.15, self.formants_sum * 0.85], voices=2)

        # --- 3) Subtle Pitch & Amplitude Drift ---
        self.slow_pitch_drift = Sine(freq=0.07, mul=0.002)  # ~0.2% drift
        self.slow_amp_drift = Sine(freq=0.05, mul=0.05)     # ~5% amplitude drift

        # --- 4) Double Chorus for Ensemble ---
        self.chorus1 = Chorus(self.colored_mix, depth=[1.4,1.5], feedback=0.2, bal=0.3)
        self.chorus2 = Chorus(self.chorus1, depth=[1.7,1.8], feedback=0.2, bal=0.3)

        # --- 5) Reverb (STRev) for a spacious choral environment ---
        # You can tweak revtime/cutoff/bal for different space size/brightness
        self.reverb = STRev(self.chorus2, inpos=0.5, revtime=3.0, cutoff=6000, bal=0.3)

        # --- 6) ADSR Envelope ---
        # Moderate attack and release
        self.env = Adsr(attack=1.0, decay=0.3, sustain=0.8, release=1.5, mul=1.0)

        # Final output => reverb * envelope * amplitude drift
        self.final_sound = (self.reverb * self.env) * (1.0 + self.slow_amp_drift)

        self.is_playing = False

    def update_formants_for_pitch(self, current_freq):
        """
        Dynamically choose vowel formants based on pitch:
            - < 170 Hz => "OO"
            - 170-300 Hz => "OH"
            - > 300 Hz => "AH"
        """
        if current_freq < 170:
            # OO Vowel
            target_f1 = 350
            target_f2 = 600
            target_f3 = 2400
            target_f4 = 3000
        elif current_freq < 300:
            # OH Vowel
            target_f1 = 450
            target_f2 = 850
            target_f3 = 2400
            target_f4 = 2800
        else:
            # AH Vowel
            target_f1 = 600
            target_f2 = 1000
            target_f3 = 2400
            target_f4 = 2800

        # Smoothly move each formant
        self.formant_f1.value = target_f1
        self.formant_f2.value = target_f2
        self.formant_f3.value = target_f3
        self.formant_f4.value = target_f4

    def update_pitch(self, pitch_factor):
        """
        pitch_factor: multiplier for base pitch (A3=220 Hz).
                      e.g., pitch_factor=2 => 440 Hz (A4).
        """
        new_freq = self.base_freq * pitch_factor * (1.0 + self.slow_pitch_drift)

        # Update oscillator frequencies
        self.fundamental.freq = new_freq
        self.fundamental_detuned1.freq = new_freq * 0.99
        self.fundamental_detuned2.freq = new_freq * 1.01

        # Harmonics
        self.harmonic1.freq = new_freq * 2
        self.harmonic2.freq = new_freq * 3
        self.harmonic3.freq = new_freq * 4

        # Vibrato
        self.vibrato_osc.freq = new_freq + self.vibrato

        # Update formants for this pitch
        self.update_formants_for_pitch(new_freq)

    def start(self):
        if not self.is_playing:
            self.final_sound.out()
            self.env.play()
            self.is_playing = True

    def stop(self):
        if self.is_playing:
            self.env.stop()
            self.is_playing = False

# ====== Instantiate ChoirVoice ======
print("Creating choir voice...")
voice = ChoirVoice(base_freq=220.0)  # A3

# ====== Webcam Feed ======
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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # Track whether any hands are detected
        hands_detected = False

        if result.multi_hand_landmarks:
            hands_detected = True
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, 
                result.multi_handedness
            ):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the current hand is labeled 'Right'
                if handedness.classification[0].label == 'Right':
                    # Get wrist Y position
                    right_hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h

                    # Example pitch mapping => [0.5, 2.0] multiplier
                    pitch_factor = 0.5 + (1 - right_hand_y / h) * 1.5
                    voice.update_pitch(pitch_factor)

                    # Visual feedback on pitch
                    cv2.putText(
                        frame, f"Pitch: {pitch_factor:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )

        # Control sound based on hand presence
        if hands_detected:
            voice.start()
            cv2.putText(
                frame, "Sound ON", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        else:
            voice.stop()
            cv2.putText(
                frame, "Sound OFF", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow("Hand Gesture Choir", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        break

print("Cleaning up...")
voice.stop()
cap.release()
cv2.destroyAllWindows()
s.stop()
print("Program ended")
