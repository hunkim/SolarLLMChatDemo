import os
import sys
import re
from typing import Dict, List, Any
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import streamlit as st
import json
from langchain_upstage import ChatUpstage 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import urllib.parse
from tinydb import TinyDB, Query
from datetime import datetime, timedelta
import hashlib
import time




def search(keyword: str, prompt: str="") -> Dict[str, Any]:
    if prompt == "":
        prompt = """Fact check the following claim and provide a structured analysis.
        
        Guidelines:
        1. Search the web for relevant information about the claim
        2. Evaluate if the claim is TRUE, FALSE, or UNCERTAIN based on available evidence
        3. Provide your response in the following JSON format:
        {
            "claim": "the original claim",
            "verdict": "TRUE/FALSE/UNCERTAIN",
            "explanation": "detailed explanation of the verdict",
            "sources": [
                {
                    "title": "source title",
                    "url": "source url",
                    "snippet": "relevant quote or information"
                }
            ]
        }

        IMPORTANT: You must detect the language of the input query and respond STRICTLY in the SAME LANGUAGE.
        - If the input query is in Korean, you MUST generate Korean search queries only
        - If the input query is in English, you MUST generate English search queries only

        Note that now is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Please check the claim and search results to provide correct analysis.Think of it carefully.
        
        Analyze this claim: """

    # Initialize the Google Generative AI client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_id = "gemini-2.0-flash"

    # Configure Google Search tool
    google_search_tool = Tool(google_search=GoogleSearch())

    # Generate content with structured output requirement
    response = client.models.generate_content(
        model=model_id,
        contents=prompt + keyword,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            temperature=0.2,  # Lower temperature for more factual responses
        ),
    )

    return response

    
  
def split_claims(text: str) -> List[str]:
    """Split text into independent, self-contained claims.
    
    Args:
        text: The original text to split into claims
    
    Returns:
        List of independent claims with resolved pronouns and self-contained context
    """
    solar_pro = ChatUpstage(model="solar-pro", model_kwargs={"response_format":{"type":"json_object"}})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a claim extraction assistant. Your task is to split text into independent, self-contained claims.

            Rules:
            1. Each claim should be a single, verifiable statement
            2. Replace all pronouns with their actual subjects
            3. Ensure each claim has complete context and can stand alone
            4. Remove subjective opinions or unverifiable statements
            5. Output claims as a valid JSON array of strings
            6. IMPORTANT: You must detect the language of the input query and respond STRICTLY in the SAME LANGUAGE.
            - If the input query is in Korean, you MUST generate Korean search queries only
            - If the input query is in English, you MUST generate English search queries only
            
            Example:
            Input: "Upstage is founded in 2022 and it's CEO is Sung Kim"
            Output: {{"claims": ["Upstage is founded in 2022", "Upstage CEO is Sung Kim"]}}
            
            Input: "The company launched its product in March and they expanded to Europe in June"
            Output: {{"claims": ["The company launched the company's product in March", "The company expanded to Europe in June"]}}
            """,
        ),
        ("user", "Text: {text}\nSplit into independent claims:"),
    ])
    
    chain = prompt | solar_pro | StrOutputParser()
    result = chain.invoke({"text": text})
    
    try:
        # Parse the JSON response
        parsed_json = json.loads(result)
        # Extract claims from the JSON structure
        if isinstance(parsed_json, dict) and "claims" in parsed_json:
            return parsed_json["claims"]
        elif isinstance(parsed_json, list):
            return parsed_json
        else:
            st.warning("Unexpected response format")
            return []
    except json.JSONDecodeError:
        # Fallback: clean up the response if it's not valid JSON
        st.warning("Failed to parse JSON response")
        claims = result.strip('[]').split(',')
        claims = [claim.strip().strip('"\'') for claim in claims]
        return [c for c in claims if c]  # Filter out empty claims

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="Fact Checker", layout="wide")

    st.title("✓ Fact Checker")
    st.write("Enter a statement to verify its accuracy. We'll break it down into individual claims and check each one.")
    st.warning("⚠️ **Disclaimer**: This is an experimental tool and results may not be 100% accurate. Please verify the information independently and use the provided sources to draw your own conclusions.")
    
    # Input area with a check button
    col1, col2 = st.columns([4, 1])
    with col1:
        claim = st.text_input(
            "Enter your statement:",
            value="Upstage AI is founded in 2022 and it's CEO is Sung Kim and CTO is Elon Musk",
            key="claim_input"
        )
    with col2:
        check_button = st.button("Check Facts", type="primary")

    if check_button and claim:
        # Split claims and show them
        with st.spinner("Breaking down the statement into checkable claims..."):
            claims = split_claims(claim)
            
        st.subheader("📝 Individual Claims")
        for idx, single_claim in enumerate(claims, 1):
            st.write(f"{idx}. {single_claim}")
        
        st.subheader("🔍 Fact Check Results")
        progress_bar = st.progress(0)
        
        # Check each claim
        for idx, single_claim in enumerate(claims):
            with st.spinner(f"Checking claim {idx + 1} of {len(claims)}..."):
                result = search(single_claim)
                
                # Create an expander for each claim result
                with st.expander(f"Claim {idx + 1}: {single_claim}", expanded=True):
                    try:
                        # Extract the text content from the response
                        if hasattr(result, 'candidates') and result.candidates:
                            # Get the first candidate's text content
                            content_text = result.candidates[0].content.parts[0].text
                            # Remove the markdown code block markers if present
                            content_text = content_text.strip('`json\n')
                            result_json = json.loads(content_text)
                            
                            # Display verdict with color
                            verdict = result_json.get("verdict", "UNCERTAIN")
                            if verdict == "TRUE":
                                st.success(f"Verdict: {verdict}")
                            elif verdict == "FALSE":
                                st.error(f"Verdict: {verdict}")
                            else:
                                st.warning(f"Verdict: {verdict}")
                            
                            # Display explanation
                            st.write("**Explanation:**")
                            st.write(result_json.get("explanation", "No explanation provided"))
                            
                            # Display sources
                            if result_json.get("sources"):
                                st.write("**Sources:**")
                                for source in result_json["sources"]:
                                    st.markdown(f"""
                                    - [{source['title']}]({source['url']})
                                    > {source['snippet']}
                                    """)
                        else:
                            st.write("No valid response received")
                    except (json.JSONDecodeError, AttributeError) as e:
                        st.error(f"Error parsing response: {str(e)}")
                        st.write(result)
                
            # Update progress
            progress_bar.progress((idx + 1) / len(claims))
        
        progress_bar.empty()
        st.success("✅ Fact checking completed!")

if __name__ == "__main__":
    main()