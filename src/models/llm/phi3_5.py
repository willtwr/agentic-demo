from .base_llm import BaseLLM

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class Phi3_5(BaseLLM):
    """Microsoft Phi 3.5 model"""
    def __init__(self):
        self.build_pipe()

    def build_pipe(self) -> None:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct", 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            repetition_penalty=1.05,
            do_sample=True,
            temperature=0.55,
            return_full_text=False
        )
        self.phi3_5pipe = HuggingFacePipeline(pipeline=pipe)

    def get_pipe(self):
        return self.phi3_5pipe
