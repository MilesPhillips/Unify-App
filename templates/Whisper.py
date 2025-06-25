import whisper

model = whisper.load_model("base")
result = model.transcribe("microphone_input.wav")  # record yourself first
print(result["text"])