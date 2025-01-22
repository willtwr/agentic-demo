from models.llm.llm_factory import llm_factory

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace


class GenerateAgent:
    """Generate Agent
    To generate output based on retrieved contents.
    """
    def __init__(self, 
                 model_name: str = "phi-3.5",
                 model = None):
        self.model_name = model_name
        if model is None:
            self.build_model()
        else:
            self.model = model

        # Prompt
        self.prompt = hub.pull("rlm/rag-prompt")

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[-2].content
        print("___Question___")
        print(question)

        last_message = messages[-1]
        docs = last_message.content
        print("___Last Message___")
        print(docs)
        
        # Chain
        rag_chain = self.prompt | self.model | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [AIMessage(content=response)]}
    
    def __call__(self):
        return self.generate
