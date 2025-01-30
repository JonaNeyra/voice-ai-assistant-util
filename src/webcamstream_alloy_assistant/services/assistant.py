import time
import httpx
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from pyaudio import PyAudio, paInt16


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt: ", prompt)

        try:
            response = self.chain.invoke(
                {"prompt": prompt, "image_base64": image.decode()},
                config={"configurable": {"session_id": "unused"}},
            ).strip()

            print("Response:", response)

            if response:
                self._tts(response)

        except httpx.RemoteProtocolError:
            print("⚠️ Error: Connection closed by the server. Trying again in 2s...")
            time.sleep(2)
            self.answer(prompt, image)

    def _create_inference_chain(self, model):
        system_prompt = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 

        Be friendly and helpful. Show some personality.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                )
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

    def _tts(self, response):
        player = PyAudio().open(
            format=paInt16,
            channels=1,
            rate=24000,
            output=True,
        )

        try:
            with openai.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="alloy",
                    response_format="pcm",
                    input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    player.write(chunk)

        except httpx.RemoteProtocolError:
            print("⚠️ Error: Connection closed during audio transmission.")

        except Exception as e:
            print(f"❌ Error TTS: {e}")
