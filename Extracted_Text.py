import pyttsx3  # For text-to-speech
import os
import speech_recognition as sr
from nltk.tokenize import word_tokenize
import nltk
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence


def text_to_wav(text, output_file="Output/output.wav", rate=150, volume=1.0):
    """Converts text to a WAV audio file.

    Args:
        text: The text to be converted.
        output_file: The path to the output WAV file (default: "output.wav").
        rate: The speech rate (words per minute, default: 150).
        volume: The volume (0.0 to 1.0, default: 1.0).
    """

    try:
        engine = pyttsx3.init()  # Initialize the text-to-speech engine

        # Set properties (optional)
        engine.setProperty('rate', rate)      # Speed of speech
        engine.setProperty('volume', volume)  # Volume 
        # You can also set the voice (see below)

        # Set Voice (Optional)
        # To see available voices:
        # voices = engine.getProperty('voices')
        # for voice in voices:
        #     print(voice.id) #Use the id to set the voice

        # Example: setting a specific voice (replace with the desired voice ID)
        # engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0')  # Example Windows voice ID.  May be different on your system.

        engine.save_to_file(text, output_file)  # Save speech to WAV file
        engine.runAndWait()  # Wait for the speech to be generated

        print(f"Text converted to WAV: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
              
def wav_to_text_and_tokenize(wav_file_path):
    """Converts a WAV file to text and tokenizes it."""

    r = sr.Recognizer()

    try:
        audio = AudioSegment.from_wav(wav_file_path) #load the wav file
        chunks = split_on_silence(audio, min_silence_len=700, silence_thresh=-40)  # Adjust parameters as needed
        all_text = ""

        for i, chunk in enumerate(chunks):
            chunk_file = f"chunk_{i}.wav"  # Temporary WAV file for each chunk
            chunk.export(chunk_file, format="wav")  # Export chunk to WAV

            with sr.AudioFile(chunk_file) as source:
                audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
                print(f"Chunk {i+1} recognized: {text}")
                all_text += text + " "  # Add the recognized text to the combined text
            except sr.UnknownValueError:
                print(f"Chunk {i+1}: Could not understand audio")
            except sr.RequestError as e:
                print(f"Chunk {i+1}: Could not request results; {e}")
            finally:
                os.remove(chunk_file)  # Clean up temporary WAV file
        # 2. Convert WAV to text and tokenize
        tokens = word_tokenize(all_text)
        return tokens

    except FileNotFoundError:
        print(f"File not found: {wav_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
# Download NLTK data (if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Example usage of the text_to_wav function
text_to_convert = "This is a sample text that will be converted to a WAV file.  You can customize the rate and volume."
output_wav_filename = "Output/my_audio.wav"  # You can change the file name
text_to_wav(text_to_convert, output_wav_filename, rate=130, volume=0.9)  # Adjust rate and volume as needed


# Another example of using the text_to_wav function
text_to_convert = "Hello World!"
text_to_wav(text_to_convert)  # Uses default file name and rate and volume

# Example usage of the wav_to_text_and_tokenize function
wav_file = output_wav_filename  # Replace with your WAV file path

if os.path.exists(wav_file):
    tokens = wav_to_text_and_tokenize(wav_file)

    if tokens is not None:
        if tokens:
            print("Tokens:", tokens)

            # Stop word removal (example)
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
            print("Filtered Tokens (Stop words removed):", filtered_tokens)
        else:
            print("No text was recognized.")
    else:
        print("An error occurred during processing.")

else:
    print(f"File not found: {wav_file}")