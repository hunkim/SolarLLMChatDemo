# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st

from pydantic import BaseModel, Field

from langchain_upstage import ChatUpstage as Chat
from solar_util import initialize_solar_llm

from langchain_community.document_loaders import BraveSearchLoader


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


llm = initialize_solar_llm()
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

            If the user's query is in a specific language (e.g., Korean, Japanese, Chinese), 
            respond in the same language. Match the language of your response to the user's input language.
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

            If the user's query is in a specific language (e.g., Korean, Japanese, Chinese), 
            respond in the same language. Match the language of your response to the user's input language.

            CRITICAL - CITATION REQUIREMENTS:
            You MUST cite EVERY piece of information using [X] notation. No statement should be made without a citation.
            
            IMPORTANT: Citation and Reference Rules:
            1. EVERY sentence must end with a citation [X]
            2. Multiple citations in one sentence should be listed like [1,2,3]
            3. Always include a "References:" section at the end
            4. List all references in order
            5. Each reference must include both title and URL

            ✅ CORRECT Example:
            "Palo Alto requires residential parking permits in downtown areas [1]. The annual permit fee is $50 for residents [2], 
            and applications can be submitted online or in person at City Hall [2,3]."

            Another example in Korean:
            "서울의 인구는 약 970만 명입니다 [1]. 최근 대중교통 이용률이 증가하고 있으며 [2], 
            특히 지하철 이용객이 20% 증가했습니다 [3]."

            References:
            [1] 서울시 인구통계 2023 - https://seoul.go.kr/statistics
            [2] 서울 교통현황 보고서 - https://seoul.go.kr/transport
            [3] 대중교통 이용분석 - https://seoul.go.kr/metro

            If you cannot find a specific reference in the context, indicate this clearly 
            with "[Source not found in context]" but still try to provide the information.
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
You are a search query expansion expert. For a given query, generate related search queries that will help find comprehensive information.

IMPORTANT RULES:
1. Match the language of the expanded queries to the original query's language
2. Generate 2-3 alternative phrasings or related aspects of the query
3. Keep queries concise and search-engine friendly
4. Focus on different aspects or synonyms of the original query
5. If the query is in a non-English language (e.g., Korean, Japanese, Chinese), all expanded queries should be in that same language

Examples:

English query: "how to get parking permit in boston"
["boston residential parking permit application", "boston parking permit cost", "how to apply for boston street parking permit"]

Korean query: "서울 주차 등록하는 방법"
["서울시 주차등록증 신청", "서울 거주자 주차등록 절차", "서울시 주차허가증 발급"]

Japanese query: "東京都 運転免許 更新"
["東京都 運転免許更新手続き", "運転免許センター 更新方法", "東京 免許更新 必要書類"]

Thai query: "วิธีการขอรับหน้าที่จอดรถในกทม"
["วิธีการขอรับหน้าที่จอดรถในกทม", "วิธีการขอรับหน้าที่จอดรถในกทม", "วิธีการขอรับหน้าที่จอดรถในกทม"]

Please write in Python LIST format.

---
Context: {context}
----
History: {chat_history}
---
Original query: {query}
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
        loader = BraveSearchLoader(
            api_key=st.secrets["BRAVE_API_KEY"],
            query=or_merged_search_query, search_kwargs={"count": 3}
        )
        return loader.load()
 


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
