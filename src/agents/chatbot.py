from models.llm.llm_factory import llm_factory

from langchain_huggingface import ChatHuggingFace
from langgraph.graph import MessagesState


class ChatBot:
    def __init__(self, model_name="phi-3.5"):
        self.build_model(model_name)

    def build_model(self, model_name="phi-3.5") -> None:
        self.chat_model = ChatHuggingFace(llm=llm_factory(model_name))

    def __call__(self, state: MessagesState):
        return {"messages": [self.chat_model.invoke(state["messages"])]}
