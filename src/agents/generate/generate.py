import os
from models.llm.llm_factory import llm_factory

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
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
        sysprompt_path = "./src/agents/generate/system_prompt_summarize.txt"
        if os.path.exists(sysprompt_path):
            with open(sysprompt_path, "r") as f:
                self.prompt = ChatPromptTemplate(
                    input_variables=['context', 'question'], 
                    messages=[
                        HumanMessagePromptTemplate(
                            prompt=PromptTemplate(
                                input_variables=['context', 'question'], 
                                template=f.read()
                            )
                        )
                    ]
                )

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
            dict: The updated state
        """
        print("---GENERATE---")
        messages = state["messages"]
        print(messages)
        
        for item in reversed(messages):
            if isinstance(item, HumanMessage):
                question = item.content
                break
        
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
    
    def __call__(self, state):
        return self.generate(state)
