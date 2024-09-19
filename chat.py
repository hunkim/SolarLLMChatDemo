# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage as Chat

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

#from solar_util import prompt_engineering

MODEL_NAME = "solar-pro"
if "MODEL_NAME" in st.secrets:
    MODEL_NAME = st.secrets["MODEL_NAME"]
llm = Chat(model=MODEL_NAME, max_tokens=1024, stop={"<END>"})

st.set_page_config(page_title="Chat")
st.title("SolarLLM")


chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", """You are Solar, a smart chatbot by Upstage, loved by many people. 
         Be smart, cheerful, and fun. Give engaging answers and avoid inappropriate language.
         reply in the same language of the user query.
         Solar is now being connected with a human.
         
         Please put <END> in the end of your answer."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{user_query}"),
    ]
)



def get_response(user_query, chat_history):
    chain = chat_with_history_prompt | llm | StrOutputParser()

    response = ""
    end_token = ""
    for chunk in chain.stream(
        {
            "chat_history": chat_history,
            "user_query": user_query,
        }
    ):
        print(chunk)
        response += chunk
        end_token += chunk
        
        if "<END>" in end_token:
            response = response.split("<END>")[0]
            break
        
        # Keep only the last 5 characters to check for <END>
        # end_token = end_token[-5:]
        
        yield chunk


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

#enhance_prompt = st.toggle("Enhance prompt", True)
enhance_prompt = False
if prompt := st.chat_input("What is up?"):
    if enhance_prompt:
        with st.status("Prompt engineering..."):
            new_prompt = prompt_engineering(prompt, st.session_state.messages)
            st.write(new_prompt)

        if 'enhanced_prompt' in new_prompt:
            prompt = new_prompt['enhanced_prompt']

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(get_response(prompt, st.session_state.messages))
    st.session_state.messages.append(AIMessage(content=response))

