from .phi3_5 import Phi3_5
from .phi3 import Phi3
from .llama3_2 import llama3_2
from .gemini import Gemini
from .deepseek_r1 import DeepSeekR1
from .smollm2 import SmolLM2


# define models here
models = {
    "phi-3.5": Phi3_5,
    "phi-3": Phi3,
    "llama-3.2": llama3_2,
    "gemini": Gemini,
    "deepseek": DeepSeekR1,
    "smollm2": SmolLM2
}


def llm_factory(model_name="phi-3.5"):
    """Factory function for LLM models"""
    return models[model_name]().get_pipe()
