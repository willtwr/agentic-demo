from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from chat_model import chat_model


chatmodel = chat_model()


def chatbot(state: MessagesState):
    return {"messages": [chatmodel.invoke(state["messages"])]}


memory = MemorySaver()
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

chat_graph = graph_builder.compile(checkpointer=memory)
