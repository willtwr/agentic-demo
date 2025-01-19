from .base_llm import BaseLLM

import torch
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


class llama3_2(BaseLLM):
    def __init__(self):
        self.build_pipe()

    def build_pipe(self) -> None:
        pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.55,
            return_full_text=False
        )
        self.llama3_2pipe = HuggingFacePipeline(pipeline=pipe)

    def get_pipe(self):
        return self.llama3_2pipe
