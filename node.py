from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
import re
# Agent ai node for decide to choose summarize or qna tool
def agent_node(query: str, llm) -> dict:
    routing_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing agent. Respond with ONLY ONE WORD: "qa" or "issue"
        Rules:
        - "qa" = questions about bugs/features/feedback like general information
        example1. "What are the issues on email notification?"
        example2. "What did users say about the search bar?"

        - "issue" = questions about given an issue or bug
        example1. "When I click on page 3 of the search results, it keeps taking me back to page 1. Very frustrating!"
        example2. "I tried to delete a document, and it just disappeared without even asking if I was sure."
        example3. "There's a spelling mistake on the login page."

        DO NOT add explanations. Respond ONLY: qa OR issue"""),
        ("human", "{query}")
    ])
    
    chain = routing_prompt | llm
    response = chain.invoke({"query": query})
    route = response.content.strip().lower()
    # print(route)
    # verify answer
    if "issue" in route:
        return "summary"
    else:
        return "qa"  

# Q and A node 
def qa_tool(query: str, retriever, llm) -> dict:
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question using ONLY the provided context.

        Rules:
        - Retrieve and report ONLY the information requested
        - Include bug numbers or feedback IDs when mentioned
        - DO NOT suggest solutions or fixes
        - DO NOT describe steps to reproduce
        - DO NOT discuss severity levels
        - Keep answers concise and factual

        Context: {context}"""),
        ("human", "Question: {question}")
    ])
    qa_chain = (
        {"context":retriever, "question":RunnablePassthrough()}
        | qa_prompt
        | llm
    )
    response = qa_chain.invoke(query)
    
    return {"answer": response.content}

# Summary node
def summary_tool(query: str, retriever, llm) -> dict:
    summary_prompt = ChatPromptTemplate.from_messages([(
    "system", """Analyze the bug reports and user feedback. Extract and summarize:

    1. Reported Issues: List main bugs/issues with their bug numbers
    2. Affected Features: Components affected  
    3. Severity: Report EXACT severity from bug reports (Critical/High/Medium/Low)

    Include bug IDs (e.g., Bug #41). Report severity exactly as stated in documents.

    Context: {context}"""),
        ("human", "Summarize issues: {query}")
    ])

    summary_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | summary_prompt
        | llm
    )
    response = summary_chain.invoke(query)
    
    return {"summary": response.content}