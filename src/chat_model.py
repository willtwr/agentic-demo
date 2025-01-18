from models.llm.llm_factory import llm_factory
from langchain_huggingface import ChatHuggingFace


def chat_model(model_name="phi-3.5"):
    return ChatHuggingFace(llm=llm_factory(model_name))
