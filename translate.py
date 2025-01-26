# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage as Chat
from langchain_upstage import UpstageDocumentParseLoader
import tempfile, os
import hashlib
import json
from typing import Dict, Optional, Tuple
from langchain.schema import Document

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Document Translator Pro",
    page_icon="📚",
    initial_sidebar_state="collapsed"  # Start with collapsed sidebar for cleaner look
)

# Modern, clean CSS styling
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Header Styles */
    .app-header {
        text-align: center;
        padding: 2.5rem 0;
        margin-bottom: 2rem;
    }
    
    .app-header h1 {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .app-header p {
        color: #666;
        font-size: 1.1rem;
    }
    
    /* Upload Zone Styles */
    .upload-zone {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #1E88E5;
        background: #f1f7fe;
    }
    
    /* Translation Container Styles */
    .translation-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .translation-header {
        background: #f8f9fa;
        padding: 1rem;
        border-bottom: 1px solid #eee;
    }
    
    .translation-content {
        padding: 1.5rem;
        line-height: 1.6;
    }
    
    /* Progress Indicator Styles */
    .progress-indicator {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer Styles */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize translation model
translation_llm = Chat(model="translation-enko")

class FileCache:
    def __init__(self):
        self.cache = self._load_cache()
    
    def _get_cache_path(self):
        return ".file_cache.json"
    
    def _load_cache(self):
        try:
            if os.path.exists(self._get_cache_path()):
                with open(self._get_cache_path(), 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _save_cache(self):
        with open(self._get_cache_path(), 'w') as f:
            json.dump(self.cache, f)

    def get_cached_docs(self, filename: str, content: bytes) -> Optional[list]:
        """Check if parsed documents are in cache"""
        file_hash = hashlib.sha256(content).hexdigest()
        
        if filename in self.cache:
            cached_data = self.cache[filename]
            if cached_data['hash'] == file_hash and 'docs' in cached_data:
                return [Document(page_content=doc['page_content'], metadata=doc['metadata']) 
                       for doc in cached_data['docs']]
        return None

    def store_docs(self, filename: str, content: bytes, docs: list) -> None:
        """Store parsed documents in cache"""
        file_hash = hashlib.sha256(content).hexdigest()
        serializable_docs = [{'page_content': doc.page_content, 'metadata': doc.metadata} 
                           for doc in docs]
        
        if filename not in self.cache:
            self.cache[filename] = {}
            
        self.cache[filename].update({
            'hash': file_hash,
            'docs': serializable_docs,
            'translations': {}  # Initialize translations dict
        })
        self._save_cache()

    def get_cached_translation(self, filename: str, page_content: str) -> Optional[str]:
        """Get cached translation for a specific page content"""
        if filename in self.cache:
            page_hash = hashlib.sha256(page_content.encode()).hexdigest()
            return self.cache[filename]['translations'].get(page_hash)
        return None

    def store_translation(self, filename: str, page_content: str, translation: str) -> None:
        """Store translation for a specific page content"""
        if filename not in self.cache:
            self.cache[filename] = {'translations': {}}
        
        page_hash = hashlib.sha256(page_content.encode()).hexdigest()
        if 'translations' not in self.cache[filename]:
            self.cache[filename]['translations'] = {}
            
        self.cache[filename]['translations'][page_hash] = translation
        self._save_cache()

# Initialize session state variables at the start of the app
if 'file_cache' not in st.session_state:
    st.session_state.file_cache = FileCache()
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'translation_complete' not in st.session_state:
    st.session_state.translation_complete = False
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = None

# Main App Header
st.markdown("""
    <div class="app-header">
        <h1>Document Translator Pro (Beta)</h1>
        <p>Professional-grade document translation powered by AI</p>
        <p style="color: #ff6b6b; font-size: 0.9rem; margin-top: 10px;">⚠️ This is a temporary service and may be discontinued without prior notice.</p>
    </div>
""", unsafe_allow_html=True)

# Main content area
tab1, tab2 = st.tabs(["📤 Upload & Translate", "ℹ️ About"])

with tab1:
    # Upload Section
    # st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    
    # Add file size limit (10MB = 10 * 1024 * 1024 bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    uploaded_file = st.file_uploader(
        "Drop your PDF here or click to upload",
        type=["pdf"],  # Explicitly use list format
        help="Maximum file size: 10MB",
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Validate file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("📤 File is too large! Maximum size is 10MB.")
        elif not uploaded_file.name.lower().endswith('.pdf'):
            st.error("📤 Please upload a PDF file.")
        else:
            st.success(f"📤 File '{uploaded_file.name}' uploaded successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and not uploaded_file.name in st.session_state:
        with st.status("Processing document...", expanded=True) as status:
            try:
                file_content = uploaded_file.getvalue()
                
                # Check document cache first
                cached_docs = st.session_state.file_cache.get_cached_docs(uploaded_file.name, file_content)
                
                if cached_docs is not None:
                    status.update(label="📑 Loading from cache...")
                    st.session_state.docs = cached_docs
                    st.session_state[uploaded_file.name] = True
                    st.session_state.translation_complete = False
                    st.success("✅ Document loaded from cache")
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(file_content)
                        
                        status.update(label="📑 Analyzing document structure...")
                        layzer = UpstageDocumentParseLoader(file_path, split="page")
                        docs = layzer.load()
                        
                        # Store parsed docs in cache
                        st.session_state.file_cache.store_docs(uploaded_file.name, file_content, docs)
                        
                        st.session_state.docs = docs
                        st.session_state[uploaded_file.name] = True
                        st.session_state.translation_complete = False
                    
                    st.success("✅ Document ready for translation")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    # Translation Display
    # Collect all translations
    translations = []
    if st.session_state.docs:
        if not st.session_state.translation_complete:
            st.markdown('<div class="progress-indicator">', unsafe_allow_html=True)
            progress_bar = st.progress(0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        for i, doc in enumerate(st.session_state.docs):
            st.markdown(f'<div class="translation-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="translation-header">Page {i+1}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="translation-content">', unsafe_allow_html=True)
                st.markdown("**Original Text**")
                st.markdown(doc.page_content, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="translation-content">', unsafe_allow_html=True)
                st.markdown("**Translated Text**")
                with st.spinner(""):
                    # Check cache first
                    cached_translation = st.session_state.file_cache.get_cached_translation(
                        uploaded_file.name,
                        doc.page_content
                    )
                    
                    if cached_translation is not None:
                        translated_content = cached_translation
                    else:
                        # Translate if not in cache
                        translated_content = translation_llm.invoke(doc.page_content)
                        # Store in cache
                        st.session_state.file_cache.store_translation(
                            uploaded_file.name,
                            doc.page_content,
                            translated_content.content
                        )
                        translated_content = translated_content.content

                    translations.append(translated_content)
                st.markdown(translated_content, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if not st.session_state.translation_complete:
                progress_bar.progress((i + 1) / len(st.session_state.docs))
        
        st.session_state.translation_complete = True
        
        # Create HTML content for download
        html_content = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .translation { margin-bottom: 30px; }
                .page-number { font-weight: bold; color: #1E88E5; margin-bottom: 10px; }
                .translated-text { line-height: 1.6; }
            </style>
        </head>
        <body>
        """
        
        for i, translation in enumerate(translations):
            html_content += f"""
            <div class="translation">
                <div class="page-number">Page {i+1}</div>
                <div class="translated-text">{translation}</div>
            </div>
            """
        
        html_content += "</body></html>"
        
        # Add download button
        download_filename = os.path.splitext(uploaded_file.name)[0] + '.translated.html'
        st.download_button(
            label="📥 Download Translation as HTML",
            data=html_content,
            file_name=download_filename,
            mime="text/html"
        )

with tab2:
    st.markdown("""
        ### About Document Translator Pro (Beta)
        
        Our professional document translation service uses state-of-the-art AI technology to provide:
        
        - ⚡ Fast and accurate translations
        - 📄 Support for PDF documents
        - 🔒 Secure document handling
        - 💯 High-quality output
        
        > ⚠️ **Please Note**: This is a temporary service and may be discontinued without prior notice.
        
        ### How to Use
        
        1. Upload your PDF document using the upload tab
        2. Wait for the automatic translation process
        3. Review the side-by-side translation
        4. Download the translated document
    """)

# Footer
st.markdown("""
    <div class="app-footer">
        <p>Powered by Upstage DocParse and SolarLLM</p>
        <p><a href="https://console.upstage.ai" target="_blank">console.upstage.ai</a></p>
    </div>
""", unsafe_allow_html=True)
