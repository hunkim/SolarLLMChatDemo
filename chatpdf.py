# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage as Chat
from langchain_upstage import GroundednessCheck

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_upstage import UpstageLayoutAnalysisLoader
import tempfile, os

from langchain import hub

st.title("LangChain ChatDoc")

llm = Chat()
# https://smith.langchain.com/hub/hunkim/rag-qa-with-history
chat_with_history_prompt = hub.pull("hunkim/rag-qa-with-history")

groundedness_check = GroundednessCheck()


def get_response(user_query, chat_history):
    chain = chat_with_history_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "chat_history": chat_history,
            "question": user_query,
            "context": st.session_state.docs,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs" not in st.session_state:
    st.session_state.docs = []

with st.sidebar:
    st.header(f"Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file and not uploaded_file.name in st.session_state:
        with st.status("Processing the data ..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                st.write("Indexing your document...")
                layzer = UpstageLayoutAnalysisLoader(file_path, split="page")
                # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
                docs = layzer.load()  # or layzer.lazy_load()
                st.session_state.docs = docs
                st.write(docs)

                # processed
                st.session_state[uploaded_file.name] = True

        st.success("Ready to Chat!")


for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("What is up?", disabled=not st.session_state.docs):
    st.session_state.messages.append(
        HumanMessage(
            content=prompt,
        )
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Getting context..."):
            st.write(st.session_state.docs)
        response = st.write_stream(get_response(prompt, st.session_state.messages))
        gc_result = groundedness_check.run(
            {
                "context": f"Context:{st.session_state.docs}\n\nQuestion{prompt}",
                "query": response,
            }
        )

        if gc_result == "grounded":
            gc_mark = "✅"
            st.success("✅ Groundedness check passed!")
        else:
            gc_mark = "❌"
            st.error("❌ Groundedness check failed!")

    st.session_state.messages.append(
        AIMessage(content=f"{gc_mark} {response}"),
    )
