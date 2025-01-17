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


def get_cache_db():
    """Initialize TinyDB database for caching"""
    return TinyDB('search_cache.json')


def generate_cache_key(query: str) -> str:
    """Generate a consistent cache key for a query"""
    return hashlib.md5(query.encode()).hexdigest()


def is_cache_valid(timestamp: str, hours: int = 1) -> bool:
    """Check if cached data is still valid"""
    cached_time = datetime.fromisoformat(timestamp)
    return datetime.now() - cached_time < timedelta(hours=hours)


def search(keyword: str) -> Dict[str, Any]:
    """Perform a search using Google's Generative AI with caching"""
    # Initialize cache
    db = get_cache_db()
    cache_key = generate_cache_key(keyword)
    Entry = Query()
    
    # Check cache first
    cached_result = db.get(Entry.cache_key == cache_key)
    if cached_result and is_cache_valid(cached_result['timestamp']):
        return cached_result['data']

    # Original search logic
    try:
        # Initialize the Google Generative AI client
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
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

        # Store result in cache before returning
        cache_data = {
            'cache_key': cache_key,
            'data': {
                "summary": formatted_text,
                "sources": sources,
                "query": keyword,
                "web_search_query": metadata.web_search_queries,
            },
            'timestamp': datetime.now().isoformat()
        }
        db.upsert(cache_data, Entry.cache_key == cache_key)

        return cache_data['data']

    except Exception as error:
        print(f"Search error: {error}")
        raise Exception(str(error) or "An error occurred while processing your search")


def generate_search_query(keyword: str, results: str) -> List[str]:
    """Generate search queries with caching"""
    # Initialize cache
    db = get_cache_db()
    cache_key = generate_cache_key(f"suggestions_{keyword}")
    Entry = Query()
    
    # Check cache first
    cached_result = db.get(Entry.cache_key == cache_key)
    if cached_result and is_cache_valid(cached_result['timestamp']):
        return cached_result['data']

    # Original suggestion generation logic
    try:
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

        # Store suggestions in cache before returning
        cache_data = {
            'cache_key': cache_key,
            'data': queries,
            'timestamp': datetime.now().isoformat()
        }
        db.upsert(cache_data, Entry.cache_key == cache_key)

        return queries
    except json.JSONDecodeError:
        return [keyword]


def generate_quick_answer(keyword: str, results: str) -> str:
    """Generate a one-line quick answer with caching"""
    # Initialize cache
    db = get_cache_db()
    cache_key = generate_cache_key(f"quick_answer_{keyword}")
    Entry = Query()
    
    # Check cache first
    cached_result = db.get(Entry.cache_key == cache_key)
    if cached_result and is_cache_valid(cached_result['timestamp']):
        return cached_result['data']

    try:
        llm = ChatUpstage(model="solar-mini", model_kwargs={"response_format":{"type":"json_object"}})
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful assistant that generates concise, one-line answers based on search results.
                Always return a JSON object with a "quick_answer" string containing a direct, factual response.
                IMPORTANT: You must detect the language of the input query and respond STRICTLY in the SAME LANGUAGE.
                - If the input query is in Korean, respond in Korean
                - If the input query is in English, respond in English
                
                The answer should be:
                1. No more than 20 words
                2. Direct and informative
                3. Based on the most recent/relevant information from results
                4. In the same language as the query
                
                Example 1 (Korean):
                Input: "현재 비트코인 가격은?"
                Output: {{"quick_answer": "비트코인은 현재 67,000달러 선에서 거래되고 있습니다."}}
                
                Example 2 (English):
                Input: "What is Bitcoin's price?"
                Output: {{"quick_answer": "Bitcoin is currently trading at around $67,000."}}""",
            ),
            ("user", "User query: {keyword}\nSearch results: {results}"),
            ("user", "Generate a one-line quick answer based on the search results."),
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"keyword": keyword, "results": results})

        try:
            response_json = json.loads(response)
            quick_answer = response_json.get("quick_answer", "")
            
            # Store answer in cache
            cache_data = {
                'cache_key': cache_key,
                'data': quick_answer,
                'timestamp': datetime.now().isoformat()
            }
            db.upsert(cache_data, Entry.cache_key == cache_key)
            
            return quick_answer
        except json.JSONDecodeError:
            return ""

    except Exception as e:
        print(f"Quick answer generation error: {e}")
        return ""


def perform_search_and_display(search_query: str, is_suggestion: bool = False) -> None:
    """
    Perform search and display results in the Streamlit UI with progressive loading
    """
    # First, perform the main search
    with st.spinner("Searching... Please wait"):
        result = search(search_query)
        
    # Display main results immediately
    st.markdown(result["summary"])
    
    # Display sources if available
    if result["sources"]:
        st.markdown("### Sources")
        with st.container():
            for idx, source in enumerate(result["sources"], 1):
                combined_content = " ".join(
                    [context["text"] for context in source["contexts"]]
                )[:200] + "..."

                st.markdown(
                    f"""
                    <div class="search-result" style="margin-bottom: 12px;">
                        <span class="search-title" style="font-family: arial, sans-serif;">
                            <span style="color: #545454; margin-right: 4px;">[{idx}]</span>
                            <a href="{source['url']}" target="_blank" style="color: #1a0dab; text-decoration: none;">
                                {source['title']}
                            </a>
                        </span>
                        <span style="color: #545454; font-size: 14px; margin-left: 8px;">
                            {combined_content}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Create placeholder for quick answer
    quick_answer_placeholder = st.empty()
    
    # Create placeholder for suggested queries
    suggestions_placeholder = st.empty()
    
    # Start background tasks for quick answer and suggestions
    with st.spinner("Generating additional insights..."):
        # Generate quick answer in the background
        quick_answer = generate_quick_answer(search_query, result["summary"])
        if quick_answer:
            # Insert quick answer at the top using the placeholder
            quick_answer_placeholder.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 16px;
                    margin-bottom: 20px;
                    font-size: 1.1em;
                    color: #1a73e8;
                ">
                    {quick_answer}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Generate and display suggested queries
        suggested_queries = generate_search_query(search_query, result["summary"])[:3]
        
        # Use the placeholder to display suggestions
        with suggestions_placeholder.container():
            st.markdown("### Related Searches")
            cols = st.columns(3)
            for i, query in enumerate(suggested_queries):
                cols[i].button(
                    query,
                    key=f"suggestion_{i}",
                    on_click=lambda q=query: st.query_params.update({"q": q})
                )

    # Display web search queries in a collapsible section at the bottom
    with st.expander("Search queries used", expanded=False):
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


def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(page_title="Search Up", layout="wide")

    # Add title and subtitle
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0;'>SearchUp</h1>
        <p style='text-align: center; color: #666; font-size: 0.9em; margin-top: 0;'>
            powered by Google, Gemini, and Solar
        </p>
    """, unsafe_allow_html=True)

    # Custom CSS for a clean, Google-like UI
    st.markdown("""
        <style>
            /* Hide Streamlit header and footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            /* Center content */
            .block-container {padding-top: 2rem; padding-bottom: 2rem;}
            
            /* Search bar */
            .search-bar {
                display: flex;
                justify-content: center;
                margin-bottom: 2rem;
            }
            .search-bar input {
                width: 50%;
                padding: 0.5rem 1rem;
                border: 1px solid #dfe1e5;
                border-radius: 24px;
                font-size: 1rem;
            }
            .search-bar input:focus {
                outline: none;
                box-shadow: 0 1px 6px rgba(32,33,36,.28);
                border-color: rgba(223,225,229,0);
            }
            .search-bar button {
                background-color: #f8f9fa;
                border: 1px solid #f8f9fa;
                border-radius: 4px;
                color: #3c4043;
                font-size: 0.875rem;
                margin: 11px 4px;
                padding: 0 16px;
                line-height: 27px;
                height: 36px;
                min-width: 54px;
                text-align: center;
                cursor: pointer;
                user-select: none;
            }
            .search-bar button:hover {
                box-shadow: 0 1px 1px rgba(0,0,0,.1);
                background-color: #f8f9fa;
                border: 1px solid #dadce0;
                color: #202124;
            }
            
            /* Suggested queries */
            .suggested-queries {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                margin-top: 1rem;
            }
            
            /* Sources */
            .sources-container {
                max-height: 600px;
                overflow-y: auto;
                padding-right: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Search bar
    search_col1, search_col2 = st.columns([3,1])
    with search_col1:
        search_input = st.text_input(
            "",
            st.query_params.get("q", ""),
            placeholder="Search anything...",
            key="search_input"
        )
        
        # Check if Enter key is pressed in the search input
        if st.session_state.get("search_input"):
            if st.session_state["search_input"] != st.query_params.get("q", ""):
                st.query_params["q"] = st.session_state["search_input"]
                st.rerun()
    
    with search_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Search"):
            st.query_params["q"] = st.session_state["search_input"]
            st.rerun()

    # Only perform search if query parameter exists in URL
    if "q" in st.query_params:
        search_query = st.query_params["q"]
        if not search_query.strip():
            st.warning("Please enter a search keyword to begin.")
        else:
            perform_search_and_display(search_query)


if __name__ == "__main__":
    main()