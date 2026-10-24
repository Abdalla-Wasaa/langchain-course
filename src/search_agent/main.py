from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

load_dotenv()
tavily=TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query, num_results=3)

llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0,
)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from search-agent!")
    response = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 Ai engineering jobs using langchain in the bay area on linkedin and their details")]}
    )
    print(response)


if __name__ == "__main__":
    main()
