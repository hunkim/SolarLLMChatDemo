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


def search(keyword: str, prompt: str="") -> Dict[str, Any]:
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
 
    # Initialize the Google Generative AI client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_id = "gemini-2.0-flash-exp"

    # Configure Google Search tool
    google_search_tool = Tool(google_search=GoogleSearch())

    # Generate content
    response = client.models.generate_content(
        model=model_id,
        contents=prompt + keyword,
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
        if metadata.grounding_chunks:
            for i, chunk in enumerate(metadata.grounding_chunks):
                if chunk.web:
                    web_sources[i] = {
                        "title": chunk.web.title,
                        "url": chunk.web.uri,
                        "contexts": [],
                    }

        # st.json(metadata)

        # Add text segments to corresponding sources
        if metadata.grounding_supports:
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
        llm = ChatUpstage(model="solar-pro", model_kwargs={"response_format":{"type":"json_object"}})
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
                
                Example 1 (Korean query -> Korean response):
                Input: "User query: 현재 비트코인 가격은?\nSearch results: 비트코인이 최근 강세를 보이며 현재 67,000달러 선에서 거래되고 있습니다. 이는 작년 대비 150% 상승한 수치이며, 전문가들은 연말까지 추가 상승 가능성을 전망하고 있습니다. 특히 최근 비트코인 ETF 승인 이후 기관 투자자들의 관심이 높아지면서 가격 상승세가 지속되고 있습니다."
                Output: {{"quick_answer": "비트코인은 현재 67,000달러 선에서 거래되고 있습니다."}}
                
                Example 2 (English query -> English response):
                Input: "User query: What is Bitcoin's price?\nSearch results: Bitcoin continues its bullish trend, currently trading at around $67,000. This represents a 150% increase from last year, with experts predicting further gains by year-end. The recent approval of Bitcoin ETFs has particularly attracted institutional investors, contributing to the sustained price momentum."
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
    Perform search and display results with enhanced source list design
    """
    # CSS with improved source list styling
    st.markdown("""
        <style>
            .main .block-container {
                padding: 2rem;
                max-width: 800px;
            }
            
            .quick-answer {
                padding: 16px;
                background: #f8f9fa;
                border-left: 3px solid #1a73e8;
                margin: 16px 0;
            }
            
            .suggestion-link {
                display: block;
                padding: 8px 16px;
                background: #f8f9fa;
                border-radius: 20px;
                color: #1a73e8;
                text-align: center;
                text-decoration: none;
                margin: 8px 0;
            }
            
            .suggestion-link:hover {
                background: #e8f0fe;
            }
            
            .source-item {
                padding: 16px;
                margin: 8px 0;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                transition: background-color 0.2s ease;
            }
            
            .source-item:hover {
                background-color: #f8f9fa;
            }
            
            .source-header {
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 8px;
            }
            
            .source-number {
                color: #666;
                font-size: 0.9em;
                min-width: 24px;
            }
            
            .source-link {
                color: #1a73e8;
                text-decoration: none;
                font-weight: 500;
                flex-grow: 1;
                line-height: 1.4;
            }
            
            .source-content {
                color: #555;
                font-size: 0.9em;
                line-height: 1.5;
                margin-left: 36px;
            }
            
            h3 {
                color: #202124;
                margin: 24px 0 16px 0;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)

    web_search_query_spot = st.empty()
    summary_spot = st.empty()
    # Main search
    with st.spinner("Searching..."):
        result = search(search_query)

    # Search queries (only if there are queries)
    if result.get("web_search_query"):
        with web_search_query_spot.expander("🔍 Search queries used", expanded=False):
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
            
            for query in result["web_search_query"]:
                st.markdown(f'<div class="search-query-item">{query}</div>', unsafe_allow_html=True)

    if result["summary"]:
        st.markdown(result["summary"])

    # Quick answer (if available)
    quick_answer = generate_quick_answer(search_query, result["summary"])
    if quick_answer:
        summary_spot.markdown(
            f'<div class="quick-answer">{quick_answer}</div>',
            unsafe_allow_html=True
        )

    # Related searches (only if there are suggestions)
    suggested_queries = generate_search_query(search_query, result["summary"])
    if suggested_queries and len(suggested_queries) > 0:
        cols = st.columns(min(len(suggested_queries[:3]), 3))
        for col, query in zip(cols, suggested_queries[:3]):
            col.markdown(
                f'<a href="?q={urllib.parse.quote(query)}" class="suggestion-link">{query}</a>',
                unsafe_allow_html=True
            )

    ref_query = """For a given query and provided search results, analyze and return a JSON object containing the full list of sources.
    The output should be in the following format:
    {
        "sources": [
            {
                "url": "source URL",
                "title": "source title",
                "content": "full original content without modifications or summaries"
            }
        ]
    }
    
   
    Important: Return the content exactly as provided in the source, without summarization or modification.
    
    Query: """ + search_query 
    ref_result = search(ref_query)
    st.json(ref_result)
    # Sources with improved design
    if result.get("sources"):
        sources = [s for s in result["sources"] if s.get("title") and s.get("url")]
        if sources:
            st.markdown("### Sources")
            for idx, source in enumerate(sources, 1):
                content = " ".join([context["text"] for context in source["contexts"]])[:200] + "..."
                st.markdown(
                    f"""
                    <div class="source-item">
                        <div class="source-header">
                            <span class="source-number">{idx}</span>
                            <a href="{source['url']}" target="_blank" class="source-link">
                                {source['title']}
                            </a>
                        </div>
                        <div class="source-content">
                            {content}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
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