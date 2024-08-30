# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st

from pydantic import BaseModel, Field

from langchain_upstage import ChatUpstage as Chat

from tavily import TavilyClient


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage

MAX_TOKENS = 3000
MAX_SEAERCH_RESULTS = 2

MODEL_NAME = "solar-1-mini-chat"
if "MODEL_NAME" in st.secrets:
    MODEL_NAME = st.secrets["MODEL_NAME"]

BASE_URL = "https://api.langchain.com"
if "BASE_URL" in st.secrets:
    BASE_URL = st.secrets["BASE_URL"]

llm = Chat(model=MODEL_NAME, base_url=BASE_URL)

tavily = TavilyClient()

st.set_page_config(page_title="Search and Chat", page_icon="🔍")
st.title("LangChain ChatGPT-like clone")


chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","""You are Solar, a smart chatbot by Upstage, loved by many people. 
            Be smart, cheerful, and fun. 
            Give engaging answers from the given conetxt and avoid inappropriate language.
            If the answer is not in context, please say you don't know and ask to clarify the question.

            When you weite the answer, please cite the source like [1], [2] if possible.
            Thyen, put all the references including citation number, title, and URL at the end of the answer.
            Each reference should be in a new line.
            """,
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "Context {context}\nQuery: {user_query} "),
    ]
)

query_expansion_prompt = """
For a given query, expand it with related questions and search the web for answers.
Try to understand the purpose of the query and expand  with upto three related questions 
to privde answer to the original query. 
Note that it's for keyword-based search engines, so it should be short and concise.

Please write in json list format like this:
["number of people in France?", How many people in France?", "France population"]

Orignal query: {query}
"""

query_context_expansion_prompt = """
For a given query and context, expand it with related questions and search the web for answers.
Try to understand the purpose of the query and expand  with upto three related questions 
to privde answer to the original query. 
Note that it's for keyword-based search engines, so it should be short and concise.

Please write in json list format like this:
["number of people in France?", How many people in France?", "France population"]

Context: {context}
Orignal query: {query}
"""


# Define your desired data structure.
class List(BaseModel):
    items: list[str]


def query_expansion(query):
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=List)

    prompt = PromptTemplate(
        template=query_expansion_prompt,
        input_variables=["query"],
    )

    chain = prompt | llm | parser
    # Invoke the chain with the joke_query.

    parsed_output = chain.invoke({"query": query})

    return parsed_output


def query_context_expansion(query, context):
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=List)

    prompt = PromptTemplate(
        template=query_context_expansion_prompt,
        input_variables=["query", "context"],
    )

    chain = prompt | llm | parser
    # Invoke the chain with the joke_query.

    parsed_output = chain.invoke({"query": query, "context": context})

    return parsed_output


def get_response(user_query, context, chat_history):
    chain = chat_with_history_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "context": context,
            "chat_history": chat_history,
            "user_query": user_query,
        }
    )


def search1(query):
    with st.status("Extending query to related questions..."):
        q_list = query_expansion(query)
        st.write(q_list)

    results = []
    for q in q_list:
        with st.spinner(f"Searching for '{q}'..."):
            result = tavily.search(query=q, max_results=MAX_SEAERCH_RESULTS)["results"]
            results += result

    return results


def search2(query, context):
    with st.status("Extending query with context to related questions..."):
        q_list = query_context_expansion(query, context)
        st.write(q_list)

    results = []
    for q in q_list:
        with st.spinner(f"Searching for '{q}'..."):
            result = tavily.search(query=q, max_results=MAX_SEAERCH_RESULTS)["results"]
            results += result

    return results


def result_summary(results):
    result_summary = ""
    for r in results:
        result_summary += f"{r['title']} - {r['content']}\n"

    return result_summary


def result_reference_summary(results):
    result_summary = ""
    for i, r in enumerate(results):
        result_summary += f"[{i+1}] {r['title']} - URL: {r['url']}\n{r['content']}\n\n"

    return result_summary

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

q = "How to use residence parking permit in palo alto?"

if prompt := st.chat_input(q):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    r1 = search1(prompt)
    result1_summary = result_summary(r1)

    r2 = search2(prompt, result1_summary[:MAX_TOKENS])

    context = result_reference_summary(r1+r2)
    context = context[:MAX_TOKENS]

    with st.status("Search Results:"):
        st.write(context)

    with st.chat_message("assistant"):
        response = st.write_stream(
            get_response(prompt, context, st.session_state.messages)
        )
    st.session_state.messages.append(AIMessage(content=response))
