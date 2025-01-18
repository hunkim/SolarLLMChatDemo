# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from solar_util import initialize_solar_llm

from solar_util import prompt_engineering

import re

jai = ChatUpstage(model=st.secrets["JAI_MODEL_NAME"], base_url=st.secrets["JAI_BASE_URL"], api_key=st.secrets["JAI_API_KEY"])
solar_pro = ChatUpstage(model="solar-pro")

st.set_page_config(page_title="Chat")
st.title("SolarLLM")


def is_korean(text):
    return re.search(r'[가-힣]', text) is not None


def korean_to_thai(text):
    translate_prompt = PromptTemplate(
        template="""You are a language translator. Translate the following text from Korean to Thai.
---
Korean: {text}
---
Thai:""",
        input_variables=["text"]
    )
    chain = translate_prompt | jai | StrOutputParser()
    return chain.invoke({"text": text})

def thai_to_korean(text):
    translate_prompt = PromptTemplate(
        template="""You are a language translator. Translate the following text from Thai to Korean.
---
Thai: {text}
---
Korean:""",
        input_variables=["text"]
    )
    chain = translate_prompt | solar_pro | StrOutputParser()
    return chain.invoke({"text": text})

chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", """You are Solar, a smart chatbot by Upstage, loved by many people. 
         Be smart, cheerful, and fun. Give engaging answers and avoid inappropriate language.
         reply in the same language of the user query.
         Solar is now being connected with a human."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{user_query}"),
    ]
)



def get_response(user_query, chat_history):
    chain = chat_with_history_prompt | jai | StrOutputParser()

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
        st.markdown(message.content)

enhance_prompt = st.toggle("Enhance prompt", False)
if prompt := st.chat_input("What is up?", value="태국의 국기는 무엇이야? 무슨 색깔이지?"):
    # Check if input is Korean and translate if needed
    if is_korean(prompt):
        with st.status("Translating Korean to Thai..."):
            prompt = korean_to_thai(prompt)
            st.write(f"Translated to Thai: {prompt}")

    if enhance_prompt:
        with st.status("Prompt engineering..."):
            new_prompt = prompt_engineering(prompt, st.session_state.messages)
            st.write(new_prompt)

        if 'enhanced_prompt' in new_prompt:
            prompt = new_prompt['enhanced_prompt']

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(get_response(prompt, st.session_state.messages))
        korean_response = thai_to_korean(response)
        st.write(korean_response)


