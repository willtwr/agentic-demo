import uuid
from dotenv import load_dotenv
import gradio as gr

from agents.chatbot.chatbot import ChatBot
# from langchain_community.document_loaders import WebBaseLoader


load_dotenv()
config = {"configurable": {"thread_id": uuid.uuid4()}}

# Initialize models here so that they are not loaded more than once.
if gr.NO_RELOAD:
    # chatbot = ChatBot(model_name="gemini")
    chatbot = ChatBot(model_name="smollm2")

    # Load some documents to vector store
    # urls = [
    #     "https://lilianweng.github.io/posts/2023-06-23-agent/",
    #     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    #     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    # ]
    # docs = [WebBaseLoader(url).load() for url in urls]
    # docs_list = [item for sublist in docs for item in sublist]
    # print("----------Adding documents-----------")
    # chatbot.get_vector_store().add_documents(docs_list)
    # print("----------End adding documents-----------")


def stream_chat_graph_updates(chat_history: list):
    """Update assistant chat here"""
    for event in chatbot().stream({"messages": [("user", chat_history[-1]["content"])]}, config, stream_mode="updates"):
    # for event in chatbot().stream({"messages": [("user", chat_history[-1]["content"])]}, config, stream_mode="values"):
        print("-----------event----------------")
        print(event)

        if "tools" in event:
            message = event['tools']['messages'][-1]
            chat_history.append({"role": "assistant", "content": message.content, "metadata": {"title": f"🛠️ Used tool {message.name}"}})
        else:
            message = event[list(event.keys())[0]]['messages'][-1]
            chat_history.append({"role": "assistant", "content": message.content})

        # message = event['messages'][-1]
        # chat_history.append({"role": "assistant", "content": message.content})

        print("--------Print From Stream-----------")
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
        
        yield chat_history


def stream_user_message(message: str, chat_history: list):
    """Update user chat here and clear textbox"""
    chat_history.append({"role": "user", "content": message})
    return "", chat_history


with gr.Blocks() as demo:
    gr.Label("ChatBot")
    chat = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...", submit_btn=True)
    msg.submit(stream_user_message, [msg, chat], [msg, chat], queue=False).then(stream_chat_graph_updates, chat, chat)


if __name__ == "__main__":
    demo.launch()
