from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), top_k_results=1, doc_content_char_limit=1500)

arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(), top_k_results=1, doc_content_char_limit=1500)

duck_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(), top_k_results=1, doc_content_char_limit=1500)

def search_wikipedia(query: str) -> str:
    return wiki_tool.run(query)

def search_arxiv(query: str)-> str:
    return arxiv_tool.run(query)

def search_duckduckgo(query: str)-> str:
    return duck_tool.run(query)

tools = [wiki_tool, arxiv_tool, duck_tool]

if __name__ == "__main__":
    print("=== Wikipedia ===")
    print(search_wikipedia("Virat Kohli"))

    print("\n=== Arxiv ===")
    print(search_arxiv("Retrieval Augmented Generation"))

    print("\n=== DuckDuckGo ===")
    print(search_duckduckgo("latest OpenAI news"))

