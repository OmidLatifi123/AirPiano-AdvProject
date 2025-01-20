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
    def __init__(self, base_freq=220.0):
        """
        A synthetic “opera-like” choir voice designed to be smoother, 
        with less noise/buzz.
        """
        self.base_freq = base_freq

        # --- 1) Base Oscillators: BLIT with fewer partials to reduce buzz ---
        self.fundamental = Blit(freq=base_freq, harms=48, mul=0.3)
        self.fundamental_detuned1 = Blit(freq=base_freq * 0.99, harms=48, mul=0.2)
        self.fundamental_detuned2 = Blit(freq=base_freq * 1.01, harms=48, mul=0.2)

        # Additional harmonic partials (sine waves)
        self.harmonic1 = Sine(freq=base_freq * 2, mul=0.15)
        self.harmonic2 = Sine(freq=base_freq * 3, mul=0.1)
        self.harmonic3 = Sine(freq=base_freq * 4, mul=0.05)

        # Slight vibrato (reduce overall vibrato depth)
        self.vibrato = Sine(freq=5, mul=2)
        self.vibrato_osc = Sine(freq=base_freq + self.vibrato, mul=0.2)

        # Pink noise for breathiness (lower amplitude to reduce noise)
        self.noise = PinkNoise(mul=0.005)

        # Mix them into a stereo signal
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

        # --- 2) Formant Filters (Approx. "Ah" vowel) ---
        # Reduced formant LFO amplitude from ±40 Hz to ±20 Hz
        self.formant_lfo = Sine(freq=0.06, mul=20)

        # Smooth transitions in formant center frequencies
        self.f1_sig = SigTo(value=600,  time=0.05, init=600)
        self.f2_sig = SigTo(value=1000, time=0.05, init=1000)
        self.f3_sig = SigTo(value=2400, time=0.05, init=2400)
        self.f4_sig = SigTo(value=2800, time=0.05, init=2800)

        # Slightly lower Q to reduce ringing
        self.bpf1 = Biquad(self.raw_mix, freq=self.f1_sig + self.formant_lfo, q=8, type=2)
        self.bpf2 = Biquad(self.raw_mix, freq=self.f2_sig + self.formant_lfo, q=8, type=2)
        self.bpf3 = Biquad(self.raw_mix, freq=self.f3_sig + self.formant_lfo, q=8, type=2)
        self.bpf4 = Biquad(self.raw_mix, freq=self.f4_sig + self.formant_lfo, q=8, type=2)

        # Parallel mix of formant filters
        self.formants_sum = Mix(
            [
                self.bpf1,
                self.bpf2,
                self.bpf3,
                self.bpf4
            ],
            voices=2
        )

        # More heavily filtered mix for smoother sound
        # (less raw => less buzz)
        self.colored_mix = Mix([self.raw_mix * 0.1, self.formants_sum * 0.9], voices=2)

        # --- 3) Subtle Pitch & Amplitude Drift (reduced) ---
        self.slow_pitch_drift = Sine(freq=0.07, mul=0.001)  # ~0.1% drift
        self.slow_amp_drift = Sine(freq=0.05, mul=0.03)     # ~3% amplitude drift

        # --- 4) Double Chorus (reduced feedback) ---
        self.chorus1 = Chorus(self.colored_mix, depth=[1.4,1.5], feedback=0.1, bal=0.3)
        self.chorus2 = Chorus(self.chorus1, depth=[1.7,1.8], feedback=0.1, bal=0.3)

        # --- 5) Stereo Reverb (shorter revtime, lower cutoff) ---
        self.reverb = STRev(self.chorus2, inpos=0.5, revtime=2.5, cutoff=5000, bal=0.3)

        # --- 6) ADSR Envelope ---
        self.env = Adsr(attack=1.2, decay=0.3, sustain=0.8, release=2.0, mul=1.0)

        # Final output => reverb * envelope * amplitude drift
        self.final_sound = (self.reverb * self.env) * (1.0 + self.slow_amp_drift)

        self.is_playing = False

    def update_pitch(self, pitch_factor):
        """
        pitch_factor: a multiplier for the base pitch (A3=220 Hz).
                      If pitch_factor=2, pitch would be 440 Hz (A4).
        """
        new_freq = self.base_freq * pitch_factor * (1.0 + self.slow_pitch_drift)

        # Update freq of fundamental and detuned oscillators
        self.fundamental.freq = new_freq
        self.fundamental_detuned1.freq = new_freq * 0.99
        self.fundamental_detuned2.freq = new_freq * 1.01

        # Harmonics
        self.harmonic1.freq = new_freq * 2
        self.harmonic2.freq = new_freq * 3
        self.harmonic3.freq = new_freq * 4

        # Vibrato
        self.vibrato_osc.freq = new_freq + self.vibrato

    def start(self):
        if not self.is_playing:
            self.final_sound.out()
            self.env.play()
            self.is_playing = True

    def stop(self):
        if self.is_playing:
            self.env.stop()
            self.is_playing = False

# ====== Create ChoirVoice Synth ======
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
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks,
                                                  result.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the current hand is labeled 'Right'
                if handedness.classification[0].label == 'Right':
                    # Get wrist Y position
                    right_hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h

                    # Limit pitch range to [1.2, 2.0] (example from your previous constraint)
                    portion = 1 - (right_hand_y / h)
                    pitch_factor = 1.2 + portion * (2.0 - 1.2)

                    # Update pitch
                    voice.update_pitch(pitch_factor)

                    # Visual feedback on pitch
                    cv2.putText(frame, f"Pitch: {pitch_factor:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

        # Control sound based on hand presence
        if hands_detected:
            voice.start()
            cv2.putText(frame, "Sound ON", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            voice.stop()
            cv2.putText(frame, "Sound OFF", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
