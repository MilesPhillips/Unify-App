import whisper

def transcribe_audio(file_path):
    """
    Transcribes an audio file using Whisper.
    """
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

if __name__ == '__main__':
    # This part is for testing the function directly
    text = transcribe_audio("microphone_input.wav")
    print("Transcribed text:")
    print(text)
