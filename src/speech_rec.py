import os
import speech_recognition as sr
import pyaudio
import wave

def record_audio(output_filename, duration=10, sample_rate=44100, channels=1, chunk_size=1024):
    """
    Records audio from the microphone and saves it to a WAV file.
    
    Args:
        output_filename (str): Path to save the recorded audio file.
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sampling rate of the audio.
        channels (int): Number of audio channels.
        chunk_size (int): Size of each audio chunk.
    """
    audio = pyaudio.PyAudio()

    # Open the audio stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")
    frames = []

    # Record audio in chunks
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    return True

def audio_to_text(audio_filename):
    """
    Converts audio from a WAV file to text using the SpeechRecognition library.
    
    Args:
        audio_filename (str): Path to the audio file.
    
    Returns:
        str: Transcribed text from the audio.
    """
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_filename) as source:
        print("Extracting text from audio...")
        audio_data = recognizer.record(source)

    # Recognize speech using Google Web Speech API
    try:
        text = recognizer.recognize_google(audio_data)
        print("Text extraction successful.")
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")
        return ""

if __name__ == "__main__":
    # File path to save the recorded audio
    audio_file = "C:\\Repo\\SIADs_Audio_Text_SRS\\Datasets\\audio.wav"

    # Record audio
    record_audio(audio_file, duration=5)

    # Convert audio to text
    extracted_text = audio_to_text(audio_file)
    print("Extracted Text:")
    print(extracted_text)