# import os
import uuid
from dotenv import load_dotenv
import gradio as gr

from chat_graph import ChatGraph


load_dotenv()
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
config = {"configurable": {"thread_id": uuid.uuid4()}}

# Put models here so they are not loaded more than once.
if gr.NO_RELOAD:    
    chatgraph = ChatGraph()


def stream_chat_graph_updates(chat_history: list):
    user_input = chat_history[-1]["content"]
    for event in chatgraph().stream({"messages": [("user", user_input)]}, config, stream_mode="updates"):
        chat_history.append({"role": "assistant", "content": event["chatbot"]["messages"][-1].content})
        yield chat_history


def stream_user_message(message: str, chat_history: list):
    chat_history.append({"role": "user", "content": message})
    return "", chat_history


with gr.Blocks() as demo:
    gr.Label("ChatBot")
    chat = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...", submit_btn=True)
    msg.submit(stream_user_message, [msg, chat], [msg, chat], queue=False).then(stream_chat_graph_updates, chat, chat)


if __name__ == "__main__":
    demo.launch()
