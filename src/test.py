import uuid
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from quantize import quantization_8bit_config

from dotenv import load_dotenv


load_dotenv()


memory = MemorySaver()
graph_builder = StateGraph(MessagesState)

#----------------- for Phi3--------------------------------
# llm = HuggingFacePipeline.from_model_id(
#     model_id="microsoft/Phi-3.5-mini-instruct",
#     # model_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     device=0,
#     pipeline_kwargs={
#         "max_new_tokens": 1024,
#         "do_sample": True,
#         "repetition_penalty": 1.03,
#         "top_k": 50,
#         "temperature": 0.55,
#         "return_full_text": False
#     },
#     # model_kwargs={
#     #     "quantization_config": quantization_8bit_config
#     # }
# )
#----------------- End Phi3--------------------------------

#----------------- for Phi3.5--------------------------------
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    repetition_penalty=1.05,
    do_sample=True,
    temperature=0.55,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)
#----------------- End Phi3.5--------------------------------

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
