from .base_llm import BaseLLM

from langchain_huggingface import HuggingFacePipeline


class Phi3(BaseLLM):
    """Microsoft Phi 3 model"""
    def __init__(self):
        self.build_pipe()
    
    def build_pipe(self) -> None:
        self.phi3_pipe = HuggingFacePipeline.from_model_id(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            device=0,
            pipeline_kwargs={
                "max_new_tokens": 1024,
                "do_sample": True,
                "repetition_penalty": 1.03,
                "top_k": 50,
                "temperature": 0.55,
                "return_full_text": False
            },
            # model_kwargs={
            #     "quantization_config": quantization_8bit_config
            # }
        )

    def get_pipe(self):
        return self.phi3_pipe
