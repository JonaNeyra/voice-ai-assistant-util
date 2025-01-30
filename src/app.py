import cv2
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from speech_recognition import Recognizer, Microphone, UnknownValueError

from webcamstream_alloy_assistant.services import DesktopScreenshot
from webcamstream_alloy_assistant.services import Assistant
from webcamstream_alloy_assistant.services import WebcamStream
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, stream_type.read(encode=True))

    except UnknownValueError:
        print("❌ There was an error processing the audio.")


if __name__ == "__main__":
    print("Select an option (Webcam/Desktop)")
    stream = input()
    print("Select a Model (OpenAI/GoogleGenAI): ")
    model_name = input()
    if stream == "Webcam":
        stream_type = WebcamStream().start()
    elif stream == "Desktop":
        stream_type = DesktopScreenshot().start()
    else:
        stream_type = DesktopScreenshot().start()

    if model_name == "OpenAI":
        model = ChatOpenAI(model_name="gpt-4o")
    elif model_name == "GoogleGenAI":
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    else:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    assistant = Assistant(model)
    recognizer = Recognizer()
    microphone = Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    if model_name == "OpenAI":
        while True:
            cv2.imshow("webcam", stream_type.read())
            if cv2.waitKey(1) in [27, ord("q")]:
                break

        stream_type.stop()
        cv2.destroyAllWindows()
        stop_listening(wait_for_stop=False)
    else:
        while True:
            screenshot = stream_type.read()
            if screenshot is not None:
                cv2.imshow("Desktop", screenshot)
            if cv2.waitKey(1) in [27, ord("q")]:
                break

        stream_type.stop()
        cv2.destroyAllWindows()
        stop_listening(wait_for_stop=False)
