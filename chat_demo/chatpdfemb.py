# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import (
    UpstageLayoutAnalysisLoader,
    UpstageGroundednessCheck,
    ChatUpstage,
    UpstageEmbeddings,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads

import tempfile, os

from langchain import hub

st.title("LangChain Upstage Solar ChatDoc")
st.write(
    "This is a conversational AI that can chat with you about your documents! Get your KEY at https://console.upstage.ai/"
)

llm = ChatUpstage()
# https://smith.langchain.com/hub/hunkim/rag-qa-with-history
chat_with_history_prompt = hub.pull("hunkim/rag-qa-with-history")

groundedness_check = UpstageGroundednessCheck()


def get_response(user_query, chat_history, retrieved_docs):
    chain = chat_with_history_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "chat_history": chat_history,
            "context": retrieved_docs,
            "question": user_query,
        }
    )


def query_expander(query):
    # Multi Query: Different Perspectives
    multi_query_template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {query}"""

    # RAG-Fusion: Related
    rag_fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {query} \n
    Output (4 queries):"""

    # Decomposition
    decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {query} \n
    Output (3 queries):"""

    query_expander_templates = [
        multi_query_template,
        rag_fusion_template,
        decomposition_template,
    ]

    expanded_queries = []
    for template in query_expander_templates:
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives
            | ChatUpstage(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
        expanded_queries += generate_queries.invoke({"query": query})

    return expanded_queries


def get_unique_union(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def retrieve_multiple_queries(retriever, queries):
    all_docs = []
    for query in queries:
        st.write(f"Retrieving for query: {query}")
        docs = retriever.invoke(query)
        all_docs.append(docs)

    unique_docs = get_unique_union(all_docs)
    return unique_docs


if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header(f"Add your PDF!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file and not uploaded_file.name in st.session_state:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            with st.status("Layout Analyzing ..."):
                layzer = UpstageLayoutAnalysisLoader(file_path, split="page")
                # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
                docs = layzer.load()  # or layzer.lazy_load()

                # Split
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=100
                )
                splits = text_splitter.split_documents(docs)

                st.write(f"Number of splits: {len(splits)}")

            with st.status(f"Vectorizing {len(splits)} splits ..."):
                # Embed
                vectorstore = FAISS.from_documents(
                    documents=splits, embedding=UpstageEmbeddings()
                )

                st.write("Vectorizing the document done!")

                st.session_state.retriever = vectorstore.as_retriever(k=10)

                # processed
                st.session_state[uploaded_file.name] = True

        st.success("Ready to Chat!")


for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        # if message.response_metadata.get("context"):
        #    with st.status("Got Context"):
        #        st.write(message.response_metadata.get("context"))
        st.markdown(message.content)

if prompt := st.chat_input("What is up?", disabled=not st.session_state.retriever):
    st.session_state.messages.append(
        HumanMessage(
            content=prompt,
        )
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Expending queries..."):
            expended_queries = query_expander(prompt)
            st.write(expended_queries)
        with st.status("Getting context..."):
            st.write("Retrieving...")
            retrieved_docs = retrieve_multiple_queries(
                st.session_state.retriever, expended_queries
            )
            # retrieved_docs = st.session_state.retriever.invoke(prompt)
            st.write(retrieved_docs)

        response = st.write_stream(
            get_response(prompt, st.session_state.messages, retrieved_docs)
        )
        gc_result = groundedness_check.run(
            {
                "context": f"Context:{retrieved_docs}\n\nQuestion{prompt}",
                "answer": response,
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
