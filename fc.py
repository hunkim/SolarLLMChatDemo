import streamlit as st
import os
from typing import Dict, List, Any
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import json
from langchain_upstage import ChatUpstage 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime


USE_KG = False

def gen_knowledge_graph(keyword: str) -> Dict[str, Any]:
    prompt = """Generate a comprehensive knowledge graph from relevant sources about the following topic.
        
        Guidelines:
        1. Search the web for relevant information
        2. For each relevant source, carefully analyze and extract:
           - Key entities (people, organizations, places, concepts)
           - Direct and indirect relationships between entities
           - Important attributes and properties of entities
           - Temporal information (dates, time periods, sequences)
           - Factual statements and evidence
        3. Combine and structure the information into a detailed knowledge graph
        4. Provide your response in the following JSON format:
        IMPORTANT: Your response MUST be a valid, parseable JSON object. Do not include any text outside the JSON structure.
        {
            "sources": [
                {
                    "title": "source title",
                    "url": "source url",
                    "snippet": "relevant quote or information. Do not alter the original text.",
                    "knowledge_graph": {
                        "entities": {
                            "primary": {
                                "name": "entity name",
                                "type": "person/organization/place/concept",
                                "attributes": {
                                    "key": "value",
                                    "confidence": 0.0
                                }
                            },
                            "related": [
                                {
                                    "name": "related entity name",
                                    "type": "person/organization/place/concept",
                                    "attributes": {
                                        "key": "value",
                                        "confidence": 0.0
                                    }
                                }
                            ]
                        },
                        "relationships": [
                            {
                                "subject": "entity name",
                                "predicate": "relationship type",
                                "object": "related entity name",
                                "temporal_info": {
                                    "start_date": "YYYY-MM-DD",
                                    "end_date": "YYYY-MM-DD",
                                    "is_current": true/false
                                },
                                "confidence": 0.0,
                                "source_urls": ["url1", "url2"]
                            }
                        ],
                    }
                }
            ]
        }

        IMPORTANT: 
        - Focus on creating comprehensive and accurate knowledge graphs
        - Include all relevant relationships and connections
        - Maintain the source language (Korean for Korean queries, English for English queries)
        - Verify information across multiple sources when possible
        - Include specific dates and temporal relationships
        
        Design your search query for the claim.

        Generate knowledge graph for this claim: """

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
            temperature=0.1,  # Lower temperature for more precise and factual output
            response_mime_type='application/json',  
        ),
    )

    return response

def search(keyword: str) -> Dict[str, Any]:
    prompt = f"""Do Search for this claim: {keyword} 
    
    Then, provide a structured analysis.
        
        Guidelines:
        1. Search the web for relevant information about the claim
        2. Evaluate if the claim is TRUE, FALSE, or UNCERTAIN based on available evidence
        3. Provide your response in the following JSON format:
        IMPORTANT: Your response MUST be a valid, parseable JSON object. Do not include any text outside the JSON structure.
        {{
            "claim": "{keyword}",
            "verdict": "TRUE/FALSE/UNCERTAIN", 
            "explanation": "detailed explanation of the verdict",
            "sources": [
                {{  
                    "title": "source title",
                    "url": "source url",
                    "snippet": "relevant quote or information"
                }}  
            ]
        }}

        IMPORTANT: You must detect the language of the input query and respond STRICTLY in the SAME LANGUAGE.
        - If the input query is in Korean, you MUST generate Korean search queries only
        - If the input query is in English, you MUST generate English search queries only

        Note that now is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        Please check the claim and search results to provide correct analysis. Generate detailed knowledge graphs for better fact verification.
     """

    prompt = f"""Check if this claim is true or false: {keyword} and provide a JSON output:
        {{
            "verdict": "TRUE/FALSE/UNCERTAIN", 
            "explanation": "detailed explanation of the verdict",
            "sources": [
                {{  
                    "title": "source title",
                    "url": "source url",
                    "snippet": "relevant quote or information"
                }}  
            ]
        }}
        You must detect the language of the input query and respond STRICTLY in the SAME LANGUAGE.
        """

    # Initialize the Google Generative AI client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_id = "gemini-2.0-flash"

    # Configure Google Search tool
    google_search_tool = Tool(google_search=GoogleSearch())

    # Try up to 3 times to get a valid JSON response
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    temperature=0.2,  # Lower temperature for factual responses
                ),
            )

            content_text = response.candidates[0].content.parts[0].text
            # Remove the markdown code block markers if present
            content_text = content_text.strip('`json\n')
            _ = json.loads(content_text)

            return response

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                st.warning(f"Failed to parse JSON response (attempt {attempt + 1}/{max_retries}). Retrying...")
                continue
            else:
                st.error("Failed to get valid JSON response after all attempts")
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
            if(USE_KG):
                with st.spinner(f"Generating knowledge graph for claim {idx + 1} of {len(claims)}..."):
                    knowledge_graph = gen_knowledge_graph(single_claim)
                    try:
                        if hasattr(knowledge_graph, 'candidates') and knowledge_graph.candidates:
                            # Get the first candidate's text content
                            content_text = knowledge_graph.candidates[0].content.parts[0].text
                            # Remove the markdown code block markers if present
                            content_text = content_text.strip('`json\n')
                            knowledge_graph_json = json.loads(content_text)
                            st.write("**Knowledge Graph:**")
                            st.json(knowledge_graph_json, expanded=False)  # Using st.json for better formatting
                    except (json.JSONDecodeError, AttributeError) as e:
                        st.error(f"Error parsing knowledge graph: {str(e)}")

            with st.spinner(f"Checking claim {idx + 1} of {len(claims)}..."):
                result = search(single_claim)

                with st.expander("🔍 Google queries used", expanded=False):
                    if hasattr(result.candidates[0], "grounding_metadata"):
                        metadata = result.candidates[0].grounding_metadata

                        st.markdown("""
                            <style>
                                .search-query-item {
                                    padding: 8px 12px;
                                    margin: 6px 0;
                                    background-color: #f0f2f6;
                                    border-radius: 6px;
                                    font-size: 0.9em;
                                    color: #444;
                                    border-left: 3px solid #1a73e8;
                                }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        # st.json(result)
                        # st.json(metadata)
                        # st.json(metadata.web_search_queries)

                        for query in metadata.web_search_queries:
                            st.markdown(f'<div class="search-query-item">{query}</div>', unsafe_allow_html=True)
                    
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
                            
                            # Display sources and knowledge graphs
                            if result_json.get("sources"):
                                st.write("**Sources and Knowledge Graphs:**")
                                for source in result_json["sources"]:
                                    # Display source information
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