from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace

from models.llm.llm_factory import llm_factory


class RewriteAgent:
    def __init__(self, 
                 model_name: str = "phi-3.5",
                 model = None):
        self.model_name = model_name
        if model is None:
            self.build_model()
        else:
            self.model = model

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

    def rewrite(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # Grader
        response = self.model.invoke(msg)
        return {"messages": [response]}
    
    def __call__(self):
        return self.rewrite
