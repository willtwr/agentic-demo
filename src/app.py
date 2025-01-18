import uuid
import gradio as gr

from chat_graph import chat_graph


config = {"configurable": {"thread_id": uuid.uuid4()}}


def stream_chat_graph_updates(chat_history: list):
    user_input = chat_history[-1]["content"]
    for event in chat_graph.stream({"messages": [("user", user_input)]}, config, stream_mode="updates"):
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

demo.launch()
