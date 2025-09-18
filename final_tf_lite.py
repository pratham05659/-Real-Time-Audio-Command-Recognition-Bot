import os
import queue
import numpy as np
import pyaudio # provides access to audio devices
import time
import sounddevice as sd  # play and record NumPy arrays containing audio signal
import scipy.signal
import cv2 # resizing the spectrogram.
from tensorflow import lite

# import threading
# import RPi.GPIO as GPIO

# HOME        = os.path.expanduser('~')
# RPI_HOME    = HOME + '/RPI/'
# GROK_HOME   = HOME + '/Desktop/Grok-Downloads/'
# sys.path.insert(1, RPI_HOME)
# from file_watcher import FileWatcher, device_sensor
# from grok_library import check_with_simulator,check_with_simulator2, device, sim_device, pin, GrokLib
# import threading
# grokLib = GrokLib()

# device['applicationIdentifier'] = str(os.path.splitext(os.path.basename(__file__))[0])
# device['mobile_messages'] = list()

# def simulate(list_of_sensors):
#     if list_of_sensors is not None:
#         global sim_device
#         sim_device = list_of_sensors
# def startListener1():
#     FileWatcher(simulate, 'simulation.json', RPI_HOME, 'config_file')
# thread1 = threading.Thread(target=startListener1, args=())
# thread1.daemon=True
# thread1.start()

TFLITE_MODEL_PATH = r"C:\Users\Pratham\Desktop\MINI_speech\freshmodel4.tflite"  
interpreter = lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# ---------------------------- Extract Model Details ----------------------------

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_shape = output_details[0]['shape']

LABELS = ["go", "left", "right", "stop"]  # Command Labels

# ---------------------------- Audio Parameters ----------------------------

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Match training sample rate
CHUNK = 1024
RECORD_SECONDS = 3

# ---------------------------- GPIO Motor Setup ----------------------------

# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BCM)

# in1, in2, en1 = 6, 13, 20
# in3, in4, en2 = 19, 26, 21

# GPIO.setup([in1, in2, en1, in3, in4, en2], GPIO.OUT)
# GPIO.output([in1, in2, in3, in4], GPIO.LOW)

# p1 = GPIO.PWM(en1, 1000)
# p2 = GPIO.PWM(en2, 1000)
# p1.start(40)
# p2.start(40)

def move_forward():
    # GPIO.output(in1, GPIO.HIGH)
    # GPIO.output(in2, GPIO.LOW)
    # GPIO.output(in3, GPIO.HIGH)
    # GPIO.output(in4, GPIO.LOW)
    print("BOT moving forward")

def move_left():
    # GPIO.output(in1, GPIO.LOW)
    # GPIO.output(in2, GPIO.HIGH)
    # GPIO.output(in3, GPIO.HIGH)
    # GPIO.output(in4, GPIO.LOW)
    print("BOT moving Left")

def move_right():
    # GPIO.output(in1, GPIO.HIGH)
    # GPIO.output(in2, GPIO.LOW)
    # GPIO.output(in3, GPIO.LOW)
    # GPIO.output(in4, GPIO.HIGH)
    print("BOT moving Right")

def stop():
    # GPIO.output(in1, GPIO.LOW)
    # GPIO.output(in2, GPIO.LOW)
    # GPIO.output(in3, GPIO.LOW)
    # GPIO.output(in4, GPIO.LOW)
    print("BOT stop Text ")

# ---------------------------- Audio Processing ----------------------------

def get_spectrogram(waveform):
    # Compute STFT (Short-Time Fourier Transform) using 
    f, t, Zxx = scipy.signal.stft(waveform, fs=RATE, nperseg=255, noverlap=128)
    # Compute Magnitude Spectrogram
    spectrogram = np.abs(Zxx)
    # Apply a Mel filter bank (approximate)
    mel_basis = np.linspace(0, spectrogram.shape[0], 64)
    mel_spectrogram = np.zeros((64, spectrogram.shape[1]))

    for i in range(63):
        mel_spectrogram[i, :] = np.mean(spectrogram[int(mel_basis[i]):int(mel_basis[i+1]), :], axis=0)
    
    # Resize using OpenCV to match model input
    spectrogram_resized = cv2.resize(mel_spectrogram, (64, 124))
    spectrogram_resized = np.expand_dims(spectrogram_resized, axis=-1)  # Add channel dimension

    return spectrogram_resized.astype(input_dtype)


# ---------------------------- Audio Recording ----------------------------

def record_audio():
    """Records audio for a fixed duration and preprocesses it"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("\nRecording Audio...")
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
    print("Finished Recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_data = b''.join(frames)
    waveform = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max

    if np.max(np.abs(waveform)) < 0.05:  # Silence detection
        print("Silence detected, skipping prediction.")
        return None

    return waveform

# ---------------------------- Model Prediction ----------------------------

def predict_audio():
    """Predicts command from real-time audio"""
    waveform = record_audio()
    if waveform is None:
        return None

    spectrogram = get_spectrogram(waveform)
    spectrogram = np.reshape(spectrogram, input_shape)  # Ensure correct shape

    interpreter.set_tensor(input_details[0]['index'], spectrogram)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx] * 100
    predicted_label = LABELS[class_idx]

    print(f"Predicted Command: {predicted_label} (Confidence: {confidence:.2f}%)")
    return predicted_label, confidence


# ---------------------------- Control Robot ----------------------------

def control_bot(command):
    """Moves the bot based on the recognized command"""
    if command == "go":
        print("Moving Forward")
        move_forward()
    elif command == "left":
        print("Turning Left")
        move_left()
    elif command == "right":
        print("Turning Right")
        move_right()
    elif command == "stop":
        print("Stopping")
        stop()
    else:
        print("Unknown command")
        stop()

# ---------------------------- Main Execution Loop ----------------------------

if __name__ == "__main__":
    try:
        while True:
            result = predict_audio()
            if result:
                move, confidence = result
                control_bot(move)
            else:
                print("SILENCE")
            time.sleep(2)  # Short delay before listening again
    except KeyboardInterrupt:
        print("Stopping...")
        stop()
        # GPIO.cleanup()

