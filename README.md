# -Real-Time-Audio-Command-Recognition-Bot
An offline speech recognition system using MFCC-based CNN. Captures live audio, processes spectrograms, predicts commands like 'go', 'left', 'right', and 'stop', and controls a bot in real-time with low latency and high accuracy..

# save this as project_info.py and run with python project_info.py

project_description = """
====================================================
REAL-TIME AUDIO COMMAND RECOGNITION BOT
====================================================

A real-time audio command recognition system that interprets spoken commands and controls a bot or device.
Combines audio signal processing, deep learning, and real-time execution for accurate and low-latency recognition.

Features:
---------
- Offline Speech Recognition: Works without external APIs
- High Accuracy: ~95% using MFCC-based spectrograms and CNN
- Real-Time Execution: Processes audio streams under 300ms
- Command Set: Recognizes 'go', 'left', 'right', 'stop'
- Customizable & Scalable: Easy to extend for more commands or devices
- Cross-Platform Audio Handling: Supports PyAudio and SoundDevice

Technologies Used:
------------------
- Python
- TensorFlow Lite
- PyAudio & SoundDevice
- NumPy & SciPy
- OpenCV
- Raspberry Pi GPIO (optional)

How It Works:
-------------
1. Audio Recording:
   - Captures real-time audio from microphone
   - Converts raw audio to normalized waveform
   - Detects silence to skip predictions

2. Feature Extraction:
   - Generates spectrogram via STFT
   - Converts to Mel-like features (approximate MFCC)
   - Resizes spectrogram to match CNN input

3. Command Prediction:
   - Uses pre-trained TensorFlow Lite CNN model
   - Outputs predicted command and confidence score

4. Bot Control:
   - Executes movement functions based on command
   - GPIO motor control available if deployed on hardware
"""

print(project_description)
