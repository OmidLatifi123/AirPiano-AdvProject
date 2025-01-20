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
        A synthetic “opera-like” choir voice using:
          - Rich oscillator (Blit) with a few partials
          - Pink noise for breathiness
          - Formant filtering with subtle movement
          - Double chorus
          - Stereo reverb (STRev)
          - ADSR envelope
        """
        self.base_freq = base_freq

        # --- 1) Base Oscillators: BLIT for more harmonic content ---
        self.fundamental = Blit(freq=base_freq, harms=64, mul=0.3)
        self.fundamental_detuned1 = Blit(freq=base_freq * 0.99, harms=64, mul=0.2)
        self.fundamental_detuned2 = Blit(freq=base_freq * 1.01, harms=64, mul=0.2)

        # Additional harmonic partials (sines for subtle color)
        self.harmonic1 = Sine(freq=base_freq * 2, mul=0.15)
        self.harmonic2 = Sine(freq=base_freq * 3, mul=0.1)
        self.harmonic3 = Sine(freq=base_freq * 4, mul=0.05)

        # Slight vibrato
        self.vibrato = Sine(freq=5, mul=3)
        self.vibrato_osc = Sine(freq=base_freq + self.vibrato, mul=0.2)

        # Pink noise for breathiness (very subtle)
        self.noise = PinkNoise(mul=0.02)

        # Mix them all into a stereo signal
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

        # --- 2) Formant Filters (Approx. "Ah" vowel, with subtle motion) ---
        # Lower Q to reduce metallic ringing
        self.formant_lfo = Sine(freq=0.06, mul=40)  # Slow LFO to shift formants +/- 40 Hz

        # We'll dynamically compute each formant center freq with a slight offset
        # around some typical opera "ah" formant centers: F1=600, F2=1000, etc.
        def freq_with_mod(base_f):
            return base_f + self.formant_lfo

        # Use `SigTo` so we can update frequencies smoothly each block
        self.f1_sig = SigTo(value=600, time=0.05, init=600)
        self.f2_sig = SigTo(value=1000, time=0.05, init=1000)
        self.f3_sig = SigTo(value=2400, time=0.05, init=2400)
        self.f4_sig = SigTo(value=2800, time=0.05, init=2800)

        # Bandpass filters
        self.bpf1 = Biquad(self.raw_mix, freq=self.f1_sig + self.formant_lfo, q=10, type=2)
        self.bpf2 = Biquad(self.raw_mix, freq=self.f2_sig + self.formant_lfo, q=10, type=2)
        self.bpf3 = Biquad(self.raw_mix, freq=self.f3_sig + self.formant_lfo, q=10, type=2)
        self.bpf4 = Biquad(self.raw_mix, freq=self.f4_sig + self.formant_lfo, q=10, type=2)

        # Mix filters in parallel
        self.formants_sum = Mix(
            [
                self.bpf1,
                self.bpf2,
                self.bpf3,
                self.bpf4
            ],
            voices=2
        )

        # Mix in some raw signal for brightness
        self.colored_mix = Mix([self.raw_mix * 0.15, self.formants_sum * 0.85], voices=2)

        # --- 3) Subtle Pitch & Amplitude Drift ---
        self.slow_pitch_drift = Sine(freq=0.07, mul=0.002)  # ~0.2% drift
        self.slow_amp_drift = Sine(freq=0.05, mul=0.05)     # ~5% amplitude drift

        # --- 4) Double Chorus for Ensemble ---
        # Lower feedback to reduce metallic ringing
        self.chorus1 = Chorus(self.colored_mix, depth=[1.4,1.5], feedback=0.2, bal=0.3)
        self.chorus2 = Chorus(self.chorus1, depth=[1.7,1.8], feedback=0.2, bal=0.3)

        # --- 5) Stereo Reverb (STRev) for a lush opera environment ---
        # Larger 'revtime' for bigger space
        self.reverb = STRev(self.chorus2, inpos=0.5, revtime=3.0, cutoff=6000, bal=0.3)

        # --- 6) ADSR Envelope ---
        # Opera-like: slow attack, longer release
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

                    # Limit pitch range to [1.2, 2.0]
                    # We'll map top of the frame (y=0) to pitch_factor=2.0,
                    # bottom of the frame (y=h) to pitch_factor=1.2.
                    portion = 1 - (right_hand_y / h)
                    pitch_factor = 1.2 + portion * (2.0 - 1.2)  # range: [1.2, 2.0]

                    # Update the pitch in our ChoirVoice
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
