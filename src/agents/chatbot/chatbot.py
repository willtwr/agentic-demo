from models.llm.llm_factory import llm_factory

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import MessagesState


class ChatBot:
    def __init__(self, model_name="phi-3.5"):
        self.build_model(model_name)

    def build_model(self, model_name="phi-3.5") -> None:
        llm = llm_factory(model_name)
        if not isinstance(llm, BaseChatModel):
            self.chat_model = ChatHuggingFace(llm=llm)
        else:
            self.chat_model = llm

    def bind_tools(self, tools: list):
        self.chat_model = self.chat_model.bind_tools(tools)

    def __call__(self, state: MessagesState):
        output = self.chat_model.invoke(state["messages"])
        return {"messages": [output]}
