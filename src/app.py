import cv2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from speech_recognition import Recognizer, Microphone, UnknownValueError

from webcamstream_alloy_assistant.services.assistant import Assistant
from webcamstream_alloy_assistant.services.webcam_stream import WebcamStream
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:
        print("❌ There was an error processing the audio.")


if __name__ == "__main__":
    webcam_stream = WebcamStream().start()
    # model = ChatOpenAI(model_name="gpt-4o")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    assistant = Assistant(model)
    recognizer = Recognizer()
    microphone = Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    while True:
        cv2.imshow("webcam", webcam_stream.read())
        if cv2.waitKey(1) in [27, ord("q")]:
            break

    webcam_stream.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)
