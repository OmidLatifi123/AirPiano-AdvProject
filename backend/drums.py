import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

class DrumSampler:
    def __init__(self, sample_path, hand_type):
        self.sample_rate, sample = wavfile.read(sample_path)
        if len(sample.shape) > 1:
            sample = np.mean(sample, axis=1)
        self.sample = sample.astype(np.float32) / np.max(np.abs(sample))
        
        self.position = 0
        self.is_playing = False
        self.buffer_size = 1024
        self.running = True
        
        self.hand_type = hand_type  
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.buffer_size
        )
        self.stream.start()
        self.prev_y = None
        self.velocity_threshold = 15  
        self.allow_trigger = True  
    def is_fist(self, hand_landmarks):
        # Check if fingers are curled (fist position)
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        finger_base = [5, 9, 13, 17]   # Corresponding base joints
        
        is_closed = True
        for tip, base in zip(finger_tips, finger_base):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                is_closed = False
                break
        return is_closed

    def audio_callback(self, outdata, frames, time, status):
        if self.is_playing:
            if self.position >= len(self.sample):
                self.is_playing = False
                self.position = 0
                outdata.fill(0)
                return
                
            chunk_end = min(self.position + frames, len(self.sample))
            chunk = self.sample[self.position:chunk_end]
            outdata[:len(chunk), 0] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):, 0] = 0
            self.position += frames
        else:
            outdata.fill(0)

    def trigger(self):
        self.position = 0
        self.is_playing = True

    def reset_trigger(self):
        self.allow_trigger = True

def run_drums():
    cap = cv2.VideoCapture(0)
    
    # Create DrumSampler instances for kick (right hand) and snare (left hand)
    kick = DrumSampler("sounds/Electronic-Kick-1.wav", "right")
    snare = DrumSampler("sounds/Ensoniq-ESQ-1-Snare.wav", "left")
    
    while True:
        success, image = cap.read()
        if not success:
            continue
            
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = kick.hands.process(rgb_image)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label
                
                kick.mp_draw.draw_landmarks(image, hand_landmarks, 
                                         kick.mp_hands.HAND_CONNECTIONS)
                
                y = int(hand_landmarks.landmark[8].y * image.shape[0])  
                
                if hand_label == 'Right':
                    if kick.prev_y is not None:
                        velocity = y - kick.prev_y
                        
                        if velocity > kick.velocity_threshold and kick.allow_trigger:
                            kick.trigger()
                            kick.allow_trigger = False  
                        
                        if velocity < -kick.velocity_threshold:
                            kick.reset_trigger()

                    kick.prev_y = y
                
                elif hand_label == 'Left':
                    if snare.prev_y is not None:
                        velocity = y - snare.prev_y
                        
                        if velocity > snare.velocity_threshold and snare.allow_trigger:
                            snare.trigger()
                            snare.allow_trigger = False  
                        
                        if velocity < -snare.velocity_threshold:
                            snare.reset_trigger()

                    snare.prev_y = y

        cv2.imshow('Drums', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    kick.stream.stop()
    snare.stream.stop()

if __name__ == "__main__":
    run_drums()