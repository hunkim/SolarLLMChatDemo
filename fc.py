import streamlit as st
import os
from typing import Dict, List, Any
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import json
from langchain_upstage import ChatUpstage 
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime


def get_fc(claim: str):
    fc = ChatUpstage(
        model="solar-google-fc",
        api_key=st.secrets["UPSTAGE_API_KEY"],
        base_url="https://fc.toy.x.upstage.ai/",
        model_kwargs={"stream": True},
    )


    results = []

    chain = fc | StrOutputParser()

    # Stream and accumulate responses
    for chunk in chain.stream(claim):
        if chunk:
            json_chunk = json.loads(chunk)
            results.append(json_chunk)  
            # Display the current chunk
            verdict_class = "claim-true" if json_chunk.get('verdict') == "TRUE" else (
                "claim-false" if json_chunk.get('verdict') == "FALSE" else "claim-uncertain"
            )
            
            st.markdown(f"""
                <div class='claim-container {verdict_class}'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h4 style='margin: 0;'>{json_chunk.get('claim', '')}</h4>
                        <h4 style='margin: 0; margin-left: 1rem;'>{display_verdict(json_chunk.get('verdict', ''))}</h4>
                    </div>
                    <p><strong>Analysis:</strong> {json_chunk.get('explanation', '')}</p>
                </div>
            """, unsafe_allow_html=True)
            
            if json_chunk.get('sources'):
                display_sources(json_chunk['sources'])

    return results

def display_verdict(verdict):
    if verdict == "TRUE":
        return "✅ VERIFIED"
    elif verdict == "FALSE":
        return "❌ FALSE"
    return "⚠️ UNCERTAIN"

def display_sources(sources):
    with st.expander("📚 View Sources", expanded=False):
        for source in sources:
            st.markdown("""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                    <h4><a href="{url}" target="_blank">{title}</a></h4>
                    <blockquote style='border-left: 3px solid #1f77b4; margin: 1rem 0; padding-left: 1rem;'>
                        {snippet}
                    </blockquote>
                </div>
            """.format(
                url=source['url'],
                title=source['title'],
                snippet=source['snippet']
            ), unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="Fact Checker",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .claim-container {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .claim-true {
            background-color: rgba(0, 255, 0, 0.1);
            border: 1px solid rgba(0, 255, 0, 0.2);
        }
        .claim-false {
            background-color: rgba(255, 0, 0, 0.1);
            border: 1px solid rgba(255, 0, 0, 0.2);
        }
        .claim-uncertain {
            background-color: rgba(255, 165, 0, 0.1);
            border: 1px solid rgba(255, 165, 0, 0.2);
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("✓ Solar-Google Fact Checker")
    st.markdown("""
        <p style='font-size: 1.2em; color: #666;'>
            Enter a statement to verify its accuracy. Our AI-powered system will analyze and fact-check each claim.
        </p>
    """, unsafe_allow_html=True)
    
    with st.container():
       

        # Powered by Upstage AI
        st.code("""
# Powered by Upstage AI
from langchain_upstage import ChatUpstage 
fc = ChatUpstage(
    model="solar-google-fc",
    api_key=st.secrets["UPSTAGE_API_KEY"], # Get your API key from https://console.upstage.ai/
    base_url="https://fc.toy.x.upstage.ai/,
)
                
result = fc.invoke(claim)
""", language="python")

    st.warning(
                "**Disclaimer**: This is an experimental tool and results may not be 100% accurate. "
                "Please verify the information independently and use the provided sources to draw your own conclusions.",
                icon="⚠️"
            )
    
    # Input area with a check button
    col1, col2 = st.columns([4, 1])
    with col1:
        claim = st.text_input(
            "Enter your statement:",
            value="Upstage AI is founded in 2022 and it's CEO is Sung Kim and CTO is Elon Musk",
            key="claim_input",
            placeholder="Enter a statement to fact-check..."
        )
    with col2:
        check_button = st.button("🔍 Verify Facts", type="primary", use_container_width=True)

    if check_button and claim:
        with st.spinner("🔄 Analyzing statement... Please allow a few moments while we search and verify the information"):
            result = get_fc(claim)
            
            # Show raw JSON in a collapsible section
            with st.expander("🔍 View Raw Response", expanded=False):
                st.json(result)

if __name__ == "__main__":
    main()