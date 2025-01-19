from .phi3_5 import Phi3_5Pipe
from .phi3 import Phi3Pipe


models = {
    "phi-3.5": Phi3_5Pipe,
    "phi-3": Phi3Pipe
}


def llm_factory(model_name="phi-3.5"):
    return models[model_name]().get_pipe()
