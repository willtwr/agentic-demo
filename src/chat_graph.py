from agents.chatbot import ChatBot

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver


class ChatGraph:
    def __init__(self):
        self.chatbot = ChatBot()
        self.build_graph()

    def build_graph(self) -> None:
        memory = MemorySaver()
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node("chatbot", self.chatbot)

        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

        self.chat_graph = graph_builder.compile(checkpointer=memory)

    def __call__(self):
        return self.chat_graph
