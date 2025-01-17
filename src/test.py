import uuid
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
# from quantize import quantization_8bit_config

from dotenv import load_dotenv


load_dotenv()


memory = MemorySaver()
graph_builder = StateGraph(MessagesState)

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    device=0,
    pipeline_kwargs={
        "max_new_tokens": 1024,
        "do_sample": True,
        "repetition_penalty": 1.03,
        "top_k": 50,
        "temperature": 0.55,
        "return_full_text": False
    },
    # model_kwargs={
    #     "quantization_config": quantization_8bit_config
    # }
)
llm = ChatHuggingFace(llm=llm)


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": uuid.uuid4()}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="updates"):
        print(event["chatbot"]["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Bye bye!")
            break

        stream_graph_updates(user_input)
    except:
        user_input = "Why got error?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
