# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage
from openai import OpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage


solar_mini = ChatUpstage(model="solar-mini")
deepseek_r = ChatUpstage(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com/v1",
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    max_tokens=100,
)

deepseek_r = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

st.set_page_config(page_title="Solar-Online-R", layout="wide")
st.title("Solar-Online-R")
st.caption("Deepseek-R enhanced Solar-mini: Combining Deepseek's reasoning with Solar's fast inference (Online Distillation)")



def get_reasoning(user_query, chat_history, model="deepseek-reasoner"):
    # Convert chat history to OpenAI format
    messages = [
        {
            "role": "system",
            "content": """You are Solar, a smart chatbot by Upstage, loved by many people. 
         Be smart, cheerful, and fun. Give engaging answers and avoid inappropriate language.
         reply in the same language of the user query.
         
         You will receive input in the following format:
         <reasoning>detailed analysis or reasoning about the query</reasoning>
         <user_query>the actual user question</user_query>
         
         Use the reasoning provided to give a more informed and thoughtful response to the user query.
         Focus on incorporating insights from the reasoning while maintaining a natural, conversational tone.
         Solar is now being connected with a human."""
        }
    ]
    
    # Add chat history
    for message in chat_history:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        messages.append({"role": role, "content": message.content})
    
    # Add current query
    messages.append({"role": "user", "content": user_query})
    
    response = deepseek_r.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1
    )
    if response.choices[0].message.reasoning_content:
        return response.choices[0].message.reasoning_content
    else:
        return response.choices[0].message.content


def get_response(user_query, chat_history, llm=solar_mini):
    chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Solar, a smart chatbot by Upstage, loved by many people. 
         Be smart, cheerful, and fun. Give engaging answers and avoid inappropriate language.
         reply in the same language of the user query.
         Solar is now being connected with a human.""",
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "{user_query}",
        ),
    ]
)

    chain = chat_with_history_prompt | llm | StrOutputParser()
    return chain.stream(
        {
            "chat_history": chat_history,
            "user_query": user_query,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        if role == "Human" and "<reasoning>" in message.content and "<user_query>" in message.content:
            reasoning = message.content.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            user_query = message.content.split("<user_query>")[1].split("</user_query>")[0].strip()
            with st.expander("Show reasoning"):
                st.markdown(reasoning)
            st.markdown(user_query)
        else:
            st.markdown(message.content)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Reasoning..."):
            reasoning = get_reasoning(prompt, st.session_state.messages)
            st.write(reasoning)
        prompt = f"""<reasoning>{reasoning}</reasoning>

<user_query>{prompt}</user_query>"""
        response = st.write_stream(get_response(prompt, st.session_state.messages))

    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=response))
