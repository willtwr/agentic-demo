import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from readability import Document
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults


def build_newssearch_tool():
    return DuckDuckGoSearchResults(name='news_search' ,backend='news', max_results=5)

@tool("news_search", return_direct=False)
def news_search(query: str) -> str:
    """Searches news from the internet based on query."""
    with DDGS() as ddgs:
        results = ddgs.news(query, max_results=5)

    output = []
    for result in results:
        response = requests.get(result['url'])
        soup = BeautifulSoup(response.content, 'html.parser')
        output.append(result['title'] + "\n" + soup.get_text().strip() + "\n")

        # doc = Document(response.content)
        # output.append(result['title'] + "\n" + doc.summary().strip() + "\n")

    return '\n'.join(output)
