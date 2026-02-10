from rag_pdf import load_split_pdf, retrieve_context
from agenttool import search_wikipedia, search_arxiv, search_duckduckgo
from router import route_query

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    question: str
    route : Optional[str]
    context: Optional[str]
    answer: Optional[str]

def create_router_node(llm):
    def router_node(state:AgentState):
        decision = route_query(llm, state['question'])
        return {"route": decision}
    return router_node

def pdf_node(state: AgentState):
    return {"context": retrieve_context(state["question"])}

def wiki_node(state: AgentState):
    return {"context": search_wikipedia(state["question"])}

def arxiv_node(state: AgentState):
    return {"context": search_arxiv(state["question"])}

def web_node(state: AgentState):
    return {"context": search_duckduckgo(state["question"])}

def none_node(state: AgentState):
    return {"context":None}

def create_chatbot_node(llm):
    def chatbot_node(state: AgentState):
        if state["context"]:
            prompt = f"""
            Use ONLY the context below to answer.
            If the answer is not present, say "I don't know".

            Context:
            {state["context"]}

            Question:
            {state["question"]}

            Answer:
"""
            response = llm.invoke(prompt)
        else:
            response = llm.invoke(state["question"])
        
        return {"answer": response.content}
    return chatbot_node

def build_agent_graph(llm):
    builder = StateGraph(AgentState)

    builder.add_node("router", create_router_node(llm))
    builder.add_node("pdf", pdf_node)
    builder.add_node("wiki", wiki_node)
    builder.add_node("arxiv", arxiv_node)
    builder.add_node("web", web_node)
    builder.add_node("none", none_node)
    builder.add_node("chatbot", create_chatbot_node(llm))
    
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router", lambda state: state["route"],
        {
            "PDF": "pdf",
            "WIKIPEDIA": "wiki",
            "ARXIV": "arxiv",
            "WEB": "web",
            "NONE": "none"
        }
    )

    for node in ["pdf", "wiki", "arxiv", "web", "none"]:
        builder.add_edge(node, "chatbot")

    builder.add_edge("chatbot", END)
    return builder.compile()

if __name__ == "__main__":
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()
    import os
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.6)

    graph = build_agent_graph(llm)

    print(graph.get_graph().draw_mermaid())

