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



def format_output():
    """Create color formatting functions for console output"""
    colors = {
        "blue": "\033[34m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "reset": "\033[0m",
    }

    return {
        "info": lambda text: f"{colors['blue']}{text}{colors['reset']}",
        "success": lambda text: f"{colors['green']}{text}{colors['reset']}",
        "highlight": lambda text: f"{colors['yellow']}{text}{colors['reset']}",
        "error": lambda text: f"{colors['red']}{text}{colors['reset']}",
    }


def format_response_to_markdown(text: str) -> str:
    """Format the AI response into markdown"""
    # Ensure consistent newlines
    processed_text = text.replace("\r\n", "\n")

    # Process main sections (simplified regex)
    processed_text = re.sub(
        r"^(\w[^:]+):(\s*)", r"## \1\2", processed_text, flags=re.MULTILINE
    )

    # Process sub-sections (simplified regex without look-behind)
    lines = processed_text.split("\n")
    processed_lines = []
    for line in lines:
        if re.match(r"^(\w[^:]+):(?!\d)", line):
            line = "### " + line
        processed_lines.append(line)
    processed_text = "\n".join(processed_lines)

    # Process bullet points
    processed_text = re.sub(r"^[•●○]\s*", "* ", processed_text, flags=re.MULTILINE)

    # Split into paragraphs and process
    paragraphs = [p for p in processed_text.split("\n\n") if p]
    formatted_paragraphs = []
    for p in paragraphs:
        if any(p.startswith(prefix) for prefix in ["#", "*", "-"]):
            formatted_paragraphs.append(p)
        else:
            formatted_paragraphs.append(f"{p}\n")

    return "\n\n".join(formatted_paragraphs)


def search(keyword: str) -> Dict[str, Any]:
    """Perform a search using Google's Generative AI"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables")

    try:
        # Initialize the Google Generative AI client
        client = genai.Client(api_key=api_key)
        model_id = "gemini-2.0-flash-exp"

        # Configure Google Search tool
        google_search_tool = Tool(google_search=GoogleSearch())

        # Generate content
        response = client.models.generate_content(
            model=model_id,
            contents=keyword,
            config=GenerateContentConfig(
                tools=[google_search_tool],
            ),
        )

        # Extract text from the first candidate's content
        if response.candidates and response.candidates[0].content.parts:
            text = response.candidates[0].content.parts[0].text
        else:
            raise Exception("No content found in response")

        # Extract sources from grounding metadata
        sources = []
        if hasattr(response.candidates[0], "grounding_metadata"):
            metadata = response.candidates[0].grounding_metadata

            # Create a mapping of chunk indices to web sources
            web_sources = {}
            for i, chunk in enumerate(metadata.grounding_chunks):
                if chunk.web:
                    web_sources[i] = {
                        "title": chunk.web.title,
                        "url": chunk.web.uri,
                        "contexts": [],
                    }

            # st.json(metadata)

            # Add text segments to corresponding sources
            for support in metadata.grounding_supports:
                for chunk_idx in support.grounding_chunk_indices:
                    if chunk_idx in web_sources:
                        web_sources[chunk_idx]["contexts"].append(
                            {
                                "text": support.segment.text,
                                "confidence": support.confidence_scores[0],
                            }
                        )

            # Convert to list and filter out sources with no contexts
            sources = [source for source in web_sources.values() if source["contexts"]]

        formatted_text = format_response_to_markdown(text)

        return {
            "summary": formatted_text,
            "sources": sources,
            "query": keyword,
            "web_search_query": metadata.web_search_queries,
        }

    except Exception as error:
        print(f"Search error: {error}")
        raise Exception(str(error) or "An error occurred while processing your search")


def generate_search_query(keyword: str, results: str) -> List[str]:
    llm = ChatUpstage(model="solar-mini", model_kwargs={"response_format":{"type":"json_object"}})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that generates search queries based on a user's query and the results of a previous search.
        Always return a JSON object with a "suggestions" array containing 3-5 search queries.
        IMPORTANT: You must detect the language of the input query and respond STRICTLY in the SAME LANGUAGE.
        - If the input query is in Korean, you MUST generate Korean search queries only
        - If the input query is in English, you MUST generate English search queries only
        
        Example 1 (Korean query -> Korean response):
        Input: "엔비디아 최신 뉴스"
        Output: {{"suggestions": ["엔비디아 주가 현황", "엔비디아 신제품 출시 2024", "엔비디아 AI 개발 현황", "엔비디아 최신 파트너십"]}}
        
        Example 2 (English query -> English response):
        Input: "latest nvidia news"
        Output: {{"suggestions": ["nvidia stock price today", "nvidia new product announcements 2024", "nvidia AI developments", "nvidia partnerships latest"]}}
        
        Remember: The response language MUST MATCH the input query language.""",
            ),
            ("user", "User query: {keyword}\nPrevious search results: {results}"),
            (
                "user",
                "Generate a JSON array of 3-5 new search queries that would help find more relevant information.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"keyword": keyword, "results": results})

    # Ensure the response is properly parsed as JSON and handle slicing safely
    try:
        response_json = json.loads(response)
        queries = response_json.get("suggestions", [])
        return queries if isinstance(queries, list) else [keyword]
    except json.JSONDecodeError:
        return [keyword]


def perform_search_and_display(search_query: str, is_suggestion: bool = False) -> None:
    """
    Perform search and display results in the Streamlit UI
    
    Args:
        search_query: The query to search for
        is_suggestion: Whether this search came from a suggested query
    """
    with st.spinner("Searching... Please wait"):
        result = search(search_query)

        # Display web search queries
        st.markdown(
            "<p style='color: #666; font-size: 0.8em; margin-bottom: 4px;'>Search queries used:</p>",
            unsafe_allow_html=True,
        )
        for query in result["web_search_query"]:
            st.markdown(
                f"""
                <div style="
                    background-color: #fafafa;
                    border-left: 2px solid #dddddd;
                    padding: 6px 12px;
                    margin: 2px 0;
                    border-radius: 0 2px 2px 0;
                    font-family: monospace;
                    font-size: 0.85em;
                    color: #555;
                ">
                    {query}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin: 12px 0;'></div>", unsafe_allow_html=True)

        # Display results and sources
        st.markdown(result["summary"])

        if result["sources"]:
            st.markdown("## Sources")
            # Create a scrollable container for sources
            st.markdown(
                """
                <style>
                    .sources-container {
                        max-height: 600px;  /* Shows about 3-4 sources */
                        overflow-y: auto;
                        padding-right: 10px;
                    }
                    .sources-container::-webkit-scrollbar {
                        width: 8px;
                    }
                    .sources-container::-webkit-scrollbar-track {
                        background: #f1f1f1;
                        border-radius: 10px;
                    }
                    .sources-container::-webkit-scrollbar-thumb {
                        background: #888;
                        border-radius: 10px;
                    }
                    .sources-container::-webkit-scrollbar-thumb:hover {
                        background: #555;
                    }
                </style>
                <div class="sources-container">
                """,
                unsafe_allow_html=True,
            )

            for source in result["sources"]:
                # Combine all contexts into a single string
                combined_content = "\n\n".join(
                    [context["text"] for context in source["contexts"]]
                )

                # Create container for each source
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 10px 0;
                        background-color: #f8f9fa;
                    ">
                        <h3 style="margin-top: 0;">
                            <a href="{source['url']}" target="_blank" style="text-decoration: none;">
                                {source['title']} ↗
                            </a>
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Display the markdown content in a container
                with st.container():
                    st.markdown(combined_content)

            # Close the scrollable container
            st.markdown("</div>", unsafe_allow_html=True)

    # Generate and display suggested queries
    with st.spinner("Generating suggested queries..."):
        suggested_queries = generate_search_query(search_query, result["summary"])[:3]
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color: #666; font-size: 0.8em; margin-bottom: 4px;'>Suggested searches:</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, query in enumerate(suggested_queries):
            encoded_query = urllib.parse.quote(query)  # Add URL encoding
            cols[i].markdown(
                f"""
                <div style="margin: 0.25rem;">
                    <a href="?q={encoded_query}" 
                       style="
                           display: block;
                           padding: 0.5rem 1rem;
                           background-color: #ffffff;
                           color: #0066cc;
                           border: 1px solid #0066cc;
                           border-radius: 0.25rem;
                           text-decoration: none;
                           text-align: center;
                           font-size: 0.875rem;
                           transition: all 0.2s;
                           box-sizing: border-box;
                           box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                           cursor: pointer;
                           white-space: nowrap;
                           overflow: hidden;
                           text-overflow: ellipsis;
                       "
                       onmouseover="this.style.backgroundColor='#f0f7ff'; this.style.borderColor='#004499';"
                       onmouseout="this.style.backgroundColor='#ffffff'; this.style.borderColor='#0066cc';"
                    >{query}</a>
                </div>
                """,
                unsafe_allow_html=True
            )


def main():
    """Main function to run the Streamlit app"""
    st.title("Gemini & SolarLLM Search Demo")

    # Get query parameters using the new API
    search_query = st.query_params.get("q", "")

    # Update search input with query parameter
    search_input = st.text_input("Enter your search query:", search_query or "Upstage AI 최신 제품?")

    if st.button("Search") or search_query:
        perform_search_and_display(search_input)


if __name__ == "__main__":
    main()
