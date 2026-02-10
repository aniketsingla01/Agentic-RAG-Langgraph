def route_query(llm, query)-> str:
    prompt = f"""
 
You are an intelligent AI router.    
Your task is to decide the BEST source to answer the question.

Available sources:
- PDF → questions related to an uploaded document
- WIKIPEDIA → general factual knowledge (people, places, definitions)
- ARXIV → academic or research-related questions
- Web → current or recent information
- NONE → common knowledge or reasoning you are confident about

Rules:
- Use Web for "latest", "current", "recent", or news-related questions
- Use ARXIV for research or papers
- Use WIKIPEDIA for factual lookups
- Use NONE if no external data is required
- Choose only ONE source
Use NONE if the question can be answered confidently without external information.
Do NOT use tools unnecessarily.

Return ONLY one word:
PDF | WIKIPEDIA | ARXIV | WEB | NONE

Question: {query}
"""
    return llm.invoke(prompt).content.strip().upper()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="openai/gpt-oss-120b")

    print(route_query(llm, "Who is Virat Kohli"))
    print(route_query(llm, "Latest OpenAI news"))
    print(route_query(llm, "Recent papers on RAG"))
    print(route_query(llm, "What is 2 + 2?"))
    print(route_query(llm, "Explain recursion in simple terms"))
    print(route_query(llm, "Why is the sky blue?"))
    print(route_query(llm, "What is an if-else statement in Python?"))
    print(route_query(llm, "Solve: If x = 5, what is 2x + 3?"))


