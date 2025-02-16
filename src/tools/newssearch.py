from langchain_community.tools import DuckDuckGoSearchResults


def build_newssearch_tool():
    return DuckDuckGoSearchResults(name='news_search' ,backend='news', max_results=5)
