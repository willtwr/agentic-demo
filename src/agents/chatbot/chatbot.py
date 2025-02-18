import io
import json
import uuid
from PIL import Image

from models.llm.llm_factory import llm_factory
from vector_store.chroma import ChromaVectorStore
from tools.vector_store_retriever import build_retriever_tool
from tools.weather import get_weather
from tools.newssearch import build_newssearch_tool, news_search
from tools.conditions.redirect import redirect_condition
from agents.generate.generate import GenerateAgent

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition


class ChatBot:
    """ChatBot Agent
    Able to chat and use some tools.
    """
    def __init__(self, 
                 model_name: str = "phi-3.5",
                 model = None,
                 small_model = None,
                 sys_prompt_path: str = "./src/agents/chatbot/system_prompt_name.txt"):
        self.model_name = model_name
        self.small_model = small_model
        with open(sys_prompt_path, "r") as file:
            self.sys_prompt = file.read()

        if model is None:
            self.build_model()
        else:
            self.model = model

        self.build_vector_store()
        self.build_graph()

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

    def build_vector_store(self):
        self.vector_store = ChromaVectorStore()

    def get_vector_store(self):
        return self.vector_store

    def build_graph(self) -> None:
        """Build the graph of the agent"""
        memory = MemorySaver()
        graph_builder = StateGraph(MessagesState)

        # tools
        # newssearch_tool = build_newssearch_tool()
        # vectorstore_retriever_tool = build_retriever_tool(self.vector_store.get_retriever())
        # tools = [get_weather, vectorstore_retriever_tool]
        # tools = [get_weather, newssearch_tool]
        tools = [get_weather, news_search]
        tool_node = ToolNode(tools=tools)
        self.bind_tools(tools)

        # agents as tools
        generate_agent = GenerateAgent(model=self.model if self.small_model is None else self.small_model)

        graph_builder.add_node("chatbot", self.invoke_model)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("generate", generate_agent)
        
        graph_builder.set_entry_point("chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        # graph_builder.add_edge("tools", "chatbot")
        # graph_builder.add_conditional_edges("tools", redirect_condition)
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        self.graph = graph_builder.compile(checkpointer=memory)
        print(self.graph.get_graph().draw_mermaid())
        image = Image.open(io.BytesIO(self.graph.get_graph().draw_mermaid_png()))
        image.save("demo-graph.png")

    def bind_tools(self, tools: list):
        """Bind tools for the agent to use"""
        self.model = self.model.bind_tools(tools)

    def invoke_model(self, state: MessagesState):
        """Model invoke function"""
        print("------------Invoke Model------------------")
        print(state["messages"])
        messages = [SystemMessage(self.sys_prompt)] + [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
        output = {"messages": [self.model.invoke(messages)]}
        if output["messages"][-1].content.startswith('<tool_call>'):
            contents = output["messages"][-1].content.replace('<tool_call>\n', '').replace('\n</tool_call>', '')
            contents = json.loads(contents)
            output = {"messages": [AIMessage(content="", tool_calls=[{"name": content["name"], "args": content["arguments"], "type": "tool_call", "id": str(uuid.uuid4())} for content in contents])]}

        return output

    def __call__(self):
        return self.graph
