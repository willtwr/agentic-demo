from .base_llm import BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI


class Gemini(BaseLLM):
    def __init__(self):
        self.build_pipe()

    def build_pipe(self) -> None:
        self.gemini_chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.5,
            max_tokens=1024
        )

    def get_pipe(self):
        return self.gemini_chat_model
