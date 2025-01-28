import os
from .base_llm import BaseLLM
from langchain_huggingface import HuggingFaceEndpoint


class DeepSeekR1(BaseLLM):
    """Microsoft Phi 3.5 model"""
    def __init__(self):
        self.build_pipe()

    def build_pipe(self) -> None:
        self.deepseek_r1 = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",
            max_new_tokens=1024,
            temperature=0.3,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

    def get_pipe(self):
        return self.deepseek_r1
