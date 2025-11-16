import sounddevice as sd
import soundfile as sf

def record_audio(file_path, duration=5, sample_rate=44100):
    """
    Records audio from the microphone and saves it to a file.
    """
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    sf.write(file_path, recording, sample_rate)
    print(f"Recording saved to {file_path}")

if __name__ == '__main__':
    # This part is for testing the function directly
    record_audio("test_recording.wav", duration=3)
