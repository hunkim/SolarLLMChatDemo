# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st

from pydantic import BaseModel, Field

from langchain_upstage import ChatUpstage as Chat

from langchain_community.tools import DuckDuckGoSearchResults


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage

MAX_TOKENS = 4000
MAX_SEAERCH_RESULTS = 5

MODEL_NAME = "solar-pro"
if "MODEL_NAME" in st.secrets:
    MODEL_NAME = st.secrets["MODEL_NAME"]
llm = Chat(model=MODEL_NAME)

ddg_search = DuckDuckGoSearchResults()


st.set_page_config(page_title="Search and Chat", page_icon="🔍")
st.title("SolarLLM Search")

short_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar, a smart search engine by Upstage, loved by many people. 
            
            Write one word answer if you can say "yes", "no", or direct answer. 
            Otherwise just one or two sentense short answer for the query from the given conetxt.
            Try to understand the user's intention and provide a quick answer.
            If the answer is not in context, please say you don't know and ask to clarify the question.
            """,
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """Query: {user_query} 
         ----
         Context: {context}""",
        ),
    ]
)

search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar, a smart search engine by Upstage, loved by many people. 
            
            See the origial query, context, and quick answer, and then provide detailed explanation.

            Try to understand the user's intention and provide the relevant information in detail.
            If the answer is not in context, please say you don't know and ask to clarify the question.
            Do not repeat the short answer.

            When you write the explnation, please cite the source like [1], [2] if possible.
            Thyen, put the cited references including citation number, title, and URL at the end of the answer.
            Each reference should be in a new line in the markdown format like this:

            [1] Title - URL
            [2] Title - URL
            ...
            """,
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """Query: {user_query} 
         ----
         Short answer: {short_answer}
         ----
         Context: {context}""",
        ),
    ]
)


query_context_expansion_prompt = """
For a given query and context(if provided), expand it with related questions and search the web for answers.
Try to understand the purpose of the query and expand  with upto three related questions 
to privde answer to the original query. 
Note that it's for keyword-based search engines, so it should be short and concise.

Please write in Python LIST format like this:
["number of people in France?", How many people in France?", "France population"]

---
Context: {context}
----
History: {chat_history}
---
Orignal query: {query}
"""


# Define your desired data structure.
class List(BaseModel):
    list[str]


def query_context_expansion(query, chat_history, context=None):
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=List)

    prompt = PromptTemplate(
        template=query_context_expansion_prompt,
        input_variables=["query", "context"],
    )

    chain = prompt | llm | parser
    # Invoke the chain with the joke_query.

    for attempt in range(3):
        try:
            parsed_output = chain.invoke(
                {"query": query, "chat_history": chat_history, "context": context}
            )
            return parsed_output
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed. Retrying...")

    st.error("All attempts failed. Returning empty list.")
    return []


def get_short_search(user_query, context, chat_history):
    chain = short_answer_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "context": context,
            "chat_history": chat_history,
            "user_query": user_query,
        }
    )


def get_search_desc(user_query, short_answer, context, chat_history):
    chain = search_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "context": context,
            "chat_history": chat_history,
            "user_query": user_query,
            "short_answer": short_answer,
        }
    )


def search(query, chat_history, context=None):
    with st.status("Extending query with context to related questions..."):
        q_list = query_context_expansion(query, chat_history, context)
        st.write(q_list)

    if not q_list:
        return []

    # combine all queries with "OR" operator
    or_merged_search_query = " OR ".join(q_list)
    with st.spinner(f"Searching for '{or_merged_search_query}'..."):
        results = ddg_search.invoke(or_merged_search_query)
        return results


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

    r1 = search(prompt, st.session_state.messages)
    result1_summary = str(r1)

    r2 = search(prompt, st.session_state.messages, result1_summary[:MAX_TOKENS])

    context = str(r1 + r2)
    context = context[:MAX_TOKENS]

    with st.status("Search Results:"):
        st.write(context)

    with st.chat_message("assistant"):
        short_answer = st.write_stream(
            get_short_search(prompt, context, st.session_state.messages)
        )
        desc = st.write_stream(
            get_search_desc(prompt, short_answer, context, st.session_state.messages)
        )
    st.session_state.messages.append(AIMessage(content=short_answer + desc))
