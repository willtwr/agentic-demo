import uuid
from dotenv import load_dotenv
import gradio as gr

from agents.chatbot.chatbot import ChatBot


load_dotenv()
config = {"configurable": {"thread_id": uuid.uuid4()}}

# Put models here so they are not loaded more than once.
if gr.NO_RELOAD:    
    chatbot = ChatBot(model_name="gemini")


def stream_chat_graph_updates(chat_history: list):
    messages = []
    # if len(chat_history) == 1:
    #     messages.append(("system", SYSTEM_PROMPT))
    
    user_input = chat_history[-1]["content"]
    messages.append(("user", user_input))
    for event in chatbot().stream({"messages": messages}, config, stream_mode="updates"):
        if "chatbot" in event:
            message = event['chatbot']['messages'][-1]
            chat_history.append({"role": "assistant", "content": message.content})
        elif "tools" in event:
            message = event['tools']['messages'][-1]
        
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
        
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
