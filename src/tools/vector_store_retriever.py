from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever


def build_retriever_tool(retriever: VectorStoreRetriever):
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs."
    )
    return retriever_tool
