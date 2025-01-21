from models.llm.llm_factory import llm_factory
from tools.weather import get_weather
# from tools.math import add, multiply

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools.tavily_search import TavilySearchResults


class ChatBot:
    """ChatBot Agent
    Able to chat and use some tools.
    """
    def __init__(self, 
                 model_name: str = "phi-3.5",
                 sys_prompt_path: str = "./src/agents/chatbot/system_prompt_name.txt"):
        self.model_name = model_name
        with open(sys_prompt_path, "r") as file:
            self.sys_prompt = file.read()

        self.build_model()
        self.build_graph()

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

    def build_graph(self) -> None:
        """Build the graph of the agent"""
        memory = MemorySaver()
        graph_builder = StateGraph(MessagesState)

        # search_tool = TavilySearchResults(max_results=2)
        # tools = [get_weather, search_tool]

        tools = [get_weather]
        tool_node = ToolNode(tools=tools)
        self.bind_tools(tools)

        graph_builder.add_node("chatbot", self.invoke_model)
        graph_builder.add_node("tools", tool_node)
        
        graph_builder.set_entry_point("chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")

        self.graph = graph_builder.compile(checkpointer=memory)

    def bind_tools(self, tools: list):
        """Bind tools for the agent to use"""
        self.model = self.model.bind_tools(tools)

    def invoke_model(self, state: MessagesState):
        """Model invoke function"""
        messages = [SystemMessage(self.sys_prompt)] + state["messages"]
        return {"messages": [self.model.invoke(messages)]}

    def __call__(self):
        return self.graph
