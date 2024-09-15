# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage as Chat
from langchain_upstage import UpstageLayoutAnalysisLoader


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_paste_button import paste_image_button as pbutton

import streamlit as st
import numpy as np
from PIL import Image
import base64
import io
import tempfile
import concurrent.futures


DOCV_MODEL_NAME = st.secrets["DOCV_MODEL_NAME"]
docv = Chat(model=DOCV_MODEL_NAME)


MODEL_NAME = st.secrets["MODEL_NAME"]
solar_pro = Chat(model=MODEL_NAME)

chat_with_history_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant. Answer the following questions considering the history of the conversation:
----
Chat history: {chat_history}
----
Image context in HTML from OCR: {image_context}
----
User question: {user_query}
"""
)


def get_img_context(img_bytes):
    image_context = ""
    if img_bytes:
        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(img_bytes)
            image_path = f.name

            layzer = UpstageLayoutAnalysisLoader(image_path, split="page")
            # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
            docs = layzer.load()  # or layzer.lazy_load()
            image_context = [doc.page_content for doc in docs]

    return image_context


def get_solar_pro_response(user_query, chat_history, image_context: str = None):
    chain = chat_with_history_prompt | solar_pro | StrOutputParser()

    return chain.stream(
        {
            "chat_history": chat_history,
            "image_context": image_context,
            "user_query": user_query,
        }
    )


def write_docv_response_stream(human_message):
    chain = docv | StrOutputParser()
    response = st.write_stream(
        chain.stream(st.session_state.messages + [human_message])
    )
    return response


def get_human_message(text_data, image_data=None):
    if not image_data:
        return HumanMessage(content=text_data)

    return HumanMessage(
        content=[
            {"type": "text", "text": f"{text_data}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )


def get_human_message_img_url(text_data, image_url=None):
    if not image_url:
        return HumanMessage(content=text_data)

    return HumanMessage(
        content=[
            {"type": "text", "text": f"{text_data}"},
            {
                "type": "image_url",
                "image_url": {"url": f"{image_url}"},
            },
        ],
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        if len(message.content) == 2:
            st.markdown(message.content[0]["text"])
        else:
            st.markdown(message.content)


img_file_buffer = st.file_uploader("Upload a image image", type=["png", "jpg", "jpeg"])
img_bytes = None
if img_file_buffer:
    st.image(img_file_buffer)
    img_bytes = img_file_buffer.read()

paste_result = pbutton("📋 Paste an image")
if paste_result.image_data is not None:
    st.write("Pasted image:")
    st.image(paste_result.image_data)
    img_bytes = io.BytesIO()
    paste_result.image_data.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()  # Image as bytes

if prompt := st.chat_input("What is up?"):
    human_message = get_human_message(prompt)
    if img_bytes:
        # remove the image from the buffer
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                if len(message.content) == 2:
                    if message.content[1]["type"] == "image_url":
                        st.session_state.messages.remove(message)
                        break

        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        human_message = get_human_message(prompt, img_base64)
        img_file_buffer = None

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.markdown("**Model1:**")
        response = write_docv_response_stream(human_message)

        st.markdown("**Model2:**")
        img_context = get_img_context(img_bytes)
        st.json(img_context, expanded=False)
        response2 = st.write_stream(
            get_solar_pro_response(prompt, st.session_state.messages, img_context)
        )

    st.session_state.messages.append(human_message)
    st.session_state.messages.append(AIMessage(content=response))
