# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageGroundednessCheck,
    ChatUpstage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

import tempfile, os
import zipfile

st.title("Solar HW Grader")
st.write(
    "This is Solar SNU HW grader demo. Get your KEY at https://console.upstage.ai/"
)

llm = ChatUpstage(model="solar-pro")

hw_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Prof. Solar, very nice and smart, loved by many people. 
            """,
        ),
        (
            "human",
            """For given report, please provide score 1-5 and quick summary of the report and explain your score and provide advice.
         ---
         Student report: {student_report},
         """,
        ),
    ]
)

groundedness_check = UpstageGroundednessCheck()


def get_response(retrieved_docs):
    chain = hw_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "student_report": retrieved_docs,
        }
    )


def process_pdf_file(file_path):
    with st.status(f"Document Parsing {file_path}..."):
        layzer = UpstageLayoutAnalysisLoader(file_path, split="page")
        # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
        docs = layzer.load()  # or layzer.lazy_load()

    with st.chat_message("user"):
        st.markdown(f"Grading {file_path}")

    with st.chat_message("assistant"):
        st.write_stream(get_response(docs))


uploaded_file = st.file_uploader(
    "Choose your `.pdf` or `.zip` file", type=["pdf", "zip"]
)

if uploaded_file and not uploaded_file.name in st.session_state:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.endswith(".pdf"):
            process_pdf_file(file_path)

        if uploaded_file.name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(temp_dir)

                for file in os.listdir(temp_dir):
                    if file.endswith(".pdf"):
                        pdf_path = os.path.join(temp_dir, file)
                        process_pdf_file(pdf_path)
