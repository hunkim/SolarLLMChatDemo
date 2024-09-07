# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from st_multimodal_chatinput import multimodal_chatinput

from langchain_upstage import ChatUpstage as Chat

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


MODEL_NAME = "solar-pro"

st.title("Discussion with Solar")

llm = Chat(model=MODEL_NAME)

discussion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar-Discussor, a smart discussion chatbot chatbot by Upstage, loved by many people. 
            
            You are taking about a topic and discussing with a user. Please participate in the discussion and provide engaging answers.
            If necessasy, ask for more information or clarify the question or add follow-up questions.
            Do not talk beyond the topic and do not provide inappropriate language.

            Please speak in a friendly and engaging manner. Speak shortly and clearly about 2~3 sentences. 
            Get to the point first and expand if necessary.

            Count each turn like putting [Turn n/10] at the beginning of the message. After 10 turns, please summarize the discussion and close the discussion.
            
            If you see [Turn 10/10], please do not speak.
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
        (
            "system",
            """You are Solar-Discussor, a smart discussion chatbot chatbot by Upstage, loved by many people. 
            
            By reading the discussion, summarize the discussion and provide a conclusion. 
            Use bullet points if necessary. Complement user's discussion and provide a conclusion.
            ---
            Topic: {topic}
            """,
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "Please summarize the discussion."),
    ]
)


def get_discussion(topic, discussion, chat_history, AorB):
    new_chat_history = chat_history

    if AorB:
        new_chat_history = []
        # switch human to AI and AI to human
        for chat in chat_history:
            if isinstance(chat, AIMessage):
                new_chat_history.append(HumanMessage(content=chat.content))
            else:
                new_chat_history.append(AIMessage(content=chat.content))

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

for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

topic = st.text_input("Discussion Topic")
if st.button("Start Discussion"):
    previous_discussion = ""
    for i in range(20):
        with st.chat_message("user"):
            discussion = st.write_stream(
                get_discussion(
                    topic, previous_discussion, st.session_state.messages, i % 2
                )
            )
            if discussion.startswith("[Turn 10/10]"):
                break
        with st.chat_message("assistant"):
            previous_discussion = st.write_stream(
                get_discussion(topic, discussion, st.session_state.messages, i % 2)
            )
            if previous_discussion.startswith("[Turn 10/10]"):
                break

    ## summarize the discussion
    with st.chat_message("assistant"):
        st.write_stream(get_summary(topic, st.session_state.messages))
