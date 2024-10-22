# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st

from langchain_upstage import ChatUpstage as Chat
from pydantic import BaseModel

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchResults

from solar_util import initialize_solar_llm

st.set_page_config(page_title="Discuss", page_icon="🗣️")
st.title("Self-debating Solar Pro Preview")

llm = initialize_solar_llm()

ddg_search = DuckDuckGoSearchResults()


# Define your desired data structure.
class SearchKeyword(BaseModel):
    list[str]


search_keyword_extraction = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar-Discussor, a smart discussion chatbot by Upstage, loved by many people. 
            
            You already comeup with a discussion draft.
            Now you can use google search to find more information about the discussion point.

            Please come up with 2~3 search keywords that you can use to find more information about the discussion point.
            ---
            Topic: {topic}
            """,
        ),
        (
            "human",
            """Please write search keywords in python list like ["keyword1", "keyword2", "keyword3"].
         ---
         Discusion Point draft: {discussion_candidate}

         """,
        ),
    ]
)

discussion_prompt_with_search = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar-Discussor, a smart discussion chatbot by Upstage, loved by many people. 
            
            You are taking about a topic and discussing with a user. Please participate in the discussion and provide engaging answers.
            If necessasy, ask for more information or clarify the question or add follow-up questions.
            If you find something wrong in others' discussion, correct them in a friendly manner in bold.
            Do not talk beyond the topic and do not provide inappropriate language.

            No need to agree on everything. You can have different opinions and discuss in a friendly manner.
            Find contradictions and correct them in a harsh manner.It's OK to say I don't agree with you.
            
            Speak shortly and clearly about 2~3 sentences. 
            Get to the point first and expand if necessary.

            Count each turn and put [Turn n/10] at the only beginning of your discussion only once.
            ---
            Topic: {topic}
            """,
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """Based on your ciscussion draft, we did google search. 
        Please use the search result to enhance your original discussion draft if the information is relevant and useful.
        If it is important, please add URL of the search result.
        Using all these please focus on the discussion and provide engaging answers.
        Don't thank the search result or mention the search result. Assume you already know these infomration.
        Fully Focus on the discussion with human. Discuss based on the facts and information you have.

        Please speak in a friendly and engaging manner. Speak shortly and clearly about 2~3 sentences. 
        Get to the point first and expand if necessary.

        Count each turn and put [Turn n/10] at the only beginning of your discussion only once.
        Please do only one turn discussion.

         ---
         Discusion Draft: {discussion_candidate}
         ----
         Search result: {external_information}
         """,
        ),
    ]
)


discussion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar-Discussor, a smart discussion chatbot by Upstage, loved by many people. 
            
            You are taking about a topic and discussing with a user. Please participate in the discussion and provide engaging answers.
            If necessasy, ask for more information or clarify the question or add follow-up questions.
            If you find something wrong in others' discussion, correct them in a friendly manner in bold.
            Do not talk beyond the topic and do not provide inappropriate language.

            Please speak in a friendly and engaging manner. Speak shortly and clearly about 2~3 sentences. 
            Get to the point first and expand if necessary.

            Count each turn and put [Turn n/10] at the only beginning of your discussion only once.
            Please do only one turn.

            Do not repeat the same point already mentioned. 
            ---
            Topic: {topic}
            """,
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{discussion}"),
    ]
)


summary_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """
         You are Solar-Discussor, a smart discussion chatbot  by Upstage, loved by many people. 
            
            By reading the discussion, provide comprehensive summarize of the discussion and provide a conclusion. 
            Only use previous discussion and do not add new information.
            Highlight several sentences if necessary. 
            ---
            Topic: {topic}
         ---
         Please summarize the discussion in history.""",
        ),
    ]
)


def make_human_last_in_history(chat_history):
    # No need to change if the last message is from human
    if not chat_history:
        return []

    if not isinstance(chat_history[-1], AIMessage):
        return chat_history
    
    return [
        (
            HumanMessage(content=chat.content)
            if isinstance(chat, AIMessage)
            else AIMessage(content=chat.content)
        )
        for chat in chat_history
    ]


def get_discussion_draft(topic, discussion, chat_history):
    chain = discussion_prompt | llm | StrOutputParser()
    discussion_candidate = chain.invoke(
        {
            "chat_history": chat_history,
            "topic": topic,
            "discussion": discussion,
        }
    )
    st.write(discussion_candidate)
    return discussion_candidate


def extract_search_keywords(topic, discussion_candidate):
    parser = JsonOutputParser(pydantic_object=SearchKeyword)
    keyword_chain = search_keyword_extraction | llm | parser
    try:
        search_keywords = keyword_chain.invoke(
            {
                "topic": topic,
                "format_instructions": parser.get_format_instructions(),
                "discussion_candidate": discussion_candidate,
            }
        )
        st.write(search_keywords)
        return search_keywords
    except Exception as e:
        st.error(f"Error extracting search keywords: {str(e)}")
        return []


def perform_search(search_keywords):
    if not search_keywords:
        return []

    or_merged_search_query = " OR ".join(search_keywords)
    try:
        search_results = ddg_search.invoke(or_merged_search_query, max_results=3)
        st.write(search_results)
        return search_results
    except Exception as e:
        st.error(f"Error performing search: {str(e)}")
        return []


def get_discussion(topic, discussion, chat_history, use_search=True):
    new_chat_history = make_human_last_in_history(chat_history)

    if use_search:
        with st.status("Writing discussion draft"):
            discussion_candidate = get_discussion_draft(
                topic, discussion, new_chat_history
            )

        with st.status("Extracting search keywords"):
            search_keywords = extract_search_keywords(topic, discussion_candidate)

        with st.status("Searching information"):
            search_results = perform_search(search_keywords)

        search_result_summary = str(search_results)[:3000]

        chain = discussion_prompt_with_search | llm | StrOutputParser()
        return chain.stream(
            {
                "chat_history": new_chat_history,
                "topic": topic,
                "discussion_candidate": discussion,
                "external_information": search_result_summary,
            }
        )

    chain = discussion_prompt | llm | StrOutputParser()
    return chain.stream(
        {
            "chat_history": new_chat_history,
            "topic": topic,
            "discussion": discussion,
        }
    )


def get_summary(topic, chat_history):
    chain = summary_prompt | llm | StrOutputParser()
    return chain.stream(
        {
            "chat_history": chat_history,
            "topic": topic,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if False:
    for message in st.session_state.messages:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.markdown(message.content)

topic = st.text_input("Discussion Topic", "How can I win LLM/AI hackathon?")
use_search = st.toggle("Use Search", False)
if st.button("Start Discussion"):
    st.session_state.messages = []
    previous_discussion = ""
    for i in range(5):
        with st.chat_message("user"):
            discussion = st.write_stream(
                get_discussion(
                    topic,
                    previous_discussion,
                    st.session_state.messages,
                    use_search,
                )
            )

            st.session_state.messages.append(HumanMessage(content=discussion))

            if discussion.startswith("[Turn 10/10]"):
                break
        with st.chat_message("assistant"):
            previous_discussion = st.write_stream(
                get_discussion(topic, discussion, st.session_state.messages, use_search)
            )

            st.session_state.messages.append(AIMessage(content=previous_discussion))

            if previous_discussion.startswith("[Turn 10/10]"):
                break

    ## summarize the discussion
    with st.chat_message("user"):
        st.write_stream(get_summary(topic, st.session_state.messages))
