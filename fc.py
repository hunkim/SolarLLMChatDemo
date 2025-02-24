import streamlit as st
import os
from typing import Dict, List, Any
import json
from langchain_upstage import ChatUpstage 
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

def get_fc(claim: str):
    """Process a claim using the fact-checking model and return results."""
    # Initialize the model
    fc = ChatUpstage(
        model="solar-google-fc",
        api_key=st.secrets["UPSTAGE_API_KEY"],
        base_url="https://fc.toy.x.upstage.ai/",
        model_kwargs={"stream": True},
    )

    results = []
    chain = fc | StrOutputParser()
    claim_count = 0
    
    # Create placeholder for claims list
    claims_placeholder = st.empty()
    results_container = st.container()
    
    # Stream and accumulate responses
    for idx, chunk in enumerate(chain.stream(claim)):
        if not chunk:
            continue
            
        try:
            json_chunk = json.loads(chunk)
            results.append(json_chunk)
            
            # Handle claims list
            if 'claims' in json_chunk:
                claim_count = len(json_chunk['claims'])
                claims_placeholder.markdown("### Claims to be verified:")
                claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(json_chunk['claims'])])
                claims_placeholder.markdown(claims_text)
            
            # Handle verdict display
            if 'verdict' in json_chunk:
                with results_container:
                    verdict_class = get_verdict_class(json_chunk.get('verdict', ''))
                    
                    st.markdown(f"""
                        <div class='claim-container {verdict_class}'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <h4 style='margin: 0;'>[{idx+1}/{claim_count}] {json_chunk.get('claim', '')}</h4>
                                <h4 style='margin: 0; margin-left: 1rem;'>{display_verdict(json_chunk.get('verdict', ''))}</h4>
                            </div>
                            <p><strong>Analysis:</strong> {json_chunk.get('explanation', '')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if json_chunk.get('sources'):
                        display_sources(json_chunk['sources'])
        except json.JSONDecodeError:
            st.error(f"Error parsing JSON from chunk: {chunk}")
            continue
            
    return results

def get_verdict_class(verdict: str) -> str:
    """Return the CSS class based on verdict."""
    verdict_map = {
        "TRUE": "claim-true",
        "FALSE": "claim-false"
    }
    return verdict_map.get(verdict, "claim-uncertain")

def display_verdict(verdict: str) -> str:
    """Return formatted verdict text."""
    verdict_map = {
        "TRUE": "✅ VERIFIED",
        "FALSE": "❌ FALSE"
    }
    return verdict_map.get(verdict, "⚠️ UNCERTAIN")

def display_sources(sources: List[Dict[str, str]]) -> None:
    """Display sources in an expander."""
    with st.expander("📚 View Sources", expanded=False):
        for source in sources:
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                    <h4><a href="{source.get('url', '#')}" target="_blank">{source.get('title', 'Source')}</a></h4>
                    <blockquote style='border-left: 3px solid #1f77b4; margin: 1rem 0; padding-left: 1rem;'>
                        {source.get('snippet', 'No snippet available')}
                    </blockquote>
                </div>
            """, unsafe_allow_html=True)

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
    base_url="https://fc.toy.x.upstage.ai/",
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
        claim = st.text_area(
            "Enter your statement:",
            value="Upstage AI is founded in 2022 and it's CEO is Sung Kim and CTO is Elon Musk",
            key="claim_input",
            placeholder="Enter a statement to fact-check...",
            height=100
        )
    with col2:
        check_button = st.button("🔍 Verify Facts", type="primary", use_container_width=True)

    if check_button and claim:
        with st.spinner("🔄 Analyzing statement... Please allow a few moments while we search and verify the information"):
            try:
                result = get_fc(claim)
                
                # Let's show it's done
                st.success("🔍 Analysis complete!")
                # Show raw JSON in a collapsible section
                with st.expander("🔍 View Raw Response", expanded=False):
                    st.json(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()