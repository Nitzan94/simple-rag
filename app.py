# ABOUTME: Streamlit UI for document indexing RAG pipeline
# ABOUTME: Provides file upload, chunking config, embeddings generation, and results display

import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import psycopg

from index_documents import (
    convert_file,
    chunk_text,
    generate_embedding,
    save_chunks,
    save_to_database
)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="RAG Document Indexer",
    page_icon="📚",
    layout="wide"
)

# Header with gradient background
st.markdown("""
<div style="background: linear-gradient(90deg, #4fc3f7 0%, #66bb6a 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;">
    <h1 style="color: white !important; margin: 0;">📚 RAG Document Indexer</h1>
    <p style="color: white; margin: 5px 0 0 0;">Upload documents to convert, chunk, and generate embeddings for RAG pipeline</p>
</div>
""", unsafe_allow_html=True)

# Custom CSS for light blue and green theme
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #f5f5f5;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #b3e5fc 0%, #c8e6c9 100%);
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        background-color: transparent !important;
        color: #0d47a1 !important;
        border-left: 3px solid white;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1976d2;
        font-weight: bold;
    }

    /* Headers */
    h2 {
        color: #1976d2 !important;
        padding: 10px;
        border-left: 5px solid #4fc3f7;
        background-color: #f0f9ff;
        border-radius: 5px;
    }

    h3 {
        color: #388e3c !important;
        padding: 8px;
        border-left: 4px solid #66bb6a;
    }

    /* Success boxes */
    .stSuccess {
        background-color: #c8e6c9 !important;
        border-left: 5px solid #4caf50 !important;
    }

    /* Info boxes */
    .stInfo {
        background-color: #b3e5fc !important;
        border-left: 5px solid #2196f3 !important;
    }

    /* Warning boxes */
    .stWarning {
        background-color: #fff9c4 !important;
        border-left: 5px solid #ffc107 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4fc3f7 0%, #66bb6a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #29b6f6 0%, #4caf50 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
        border-radius: 8px;
        padding: 4px;
        border: 2px solid #e0e0e0;
    }

    .stTabs [data-baseweb="tab"] {
        color: #1976d2;
        font-weight: bold;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4fc3f7;
        color: white !important;
        border-radius: 6px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        border: 2px dashed #4fc3f7;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f9ff;
        color: #1976d2;
        font-weight: bold;
        border-radius: 6px;
        border-left: 3px solid #4fc3f7;
    }

    /* Select box */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #4fc3f7;
        border-radius: 6px;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: white;
        border: 2px solid #4fc3f7;
        border-radius: 6px;
        color: #0d47a1;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #4fc3f7;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #66bb6a 0%, #4fc3f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #4caf50 0%, #29b6f6 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Chunking settings
st.sidebar.subheader("Chunking Strategy")
enable_chunking = st.sidebar.checkbox("Enable chunking", value=True)

chunk_strategy = "fixed"
chunk_size = 1000
overlap = 200

if enable_chunking:
    strategy_option = st.sidebar.radio(
        "Strategy",
        ["Fixed-size with overlap", "Sentence-based", "Paragraph-based"],
        index=0
    )

    if strategy_option == "Fixed-size with overlap":
        chunk_strategy = "fixed"
        chunk_size = st.sidebar.slider("Chunk size (chars)", 100, 5000, 1000, 100)
        overlap = st.sidebar.slider("Overlap (chars)", 0, 500, 200, 50)
    elif strategy_option == "Sentence-based":
        chunk_strategy = "sentence"
    else:
        chunk_strategy = "paragraph"

# Embedding settings
st.sidebar.subheader("Embeddings")
enable_embeddings = st.sidebar.checkbox("Generate embeddings", value=False)
api_key = None

if enable_embeddings:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        st.sidebar.success("✅ Gemini API key loaded")
    else:
        st.sidebar.error("❌ GEMINI_API_KEY not found in .env")
        enable_embeddings = False

# Database settings
st.sidebar.subheader("PostgreSQL Storage")

# Always load postgres_url for Database Browser tab
postgres_url = os.getenv("POSTGRES_URL")

enable_postgres = st.sidebar.checkbox("Save to PostgreSQL", value=False)

if enable_postgres:
    if postgres_url:
        # Test connection
        try:
            with psycopg.connect(postgres_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM document_chunks")
                    count = cur.fetchone()[0]
                st.sidebar.success(f"✅ Connected ({count} chunks in DB)")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")
            enable_postgres = False
    else:
        st.sidebar.error("❌ POSTGRES_URL not found in .env")
        enable_postgres = False

# Main tabs
tab1, tab2, tab3 = st.tabs(["📄 Single File", "📁 Batch Upload", "💾 Database Browser"])

# Tab 1: Single file upload
with tab1:
    st.header("Upload Single File")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx"],
        key="single"
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 3])

        with col1:
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.info(f"**Type:** {uploaded_file.type}")

        with col2:
            if st.button("🚀 Process File", key="process_single"):
                start_time = time.time()

                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Create local output directory
                    output_dir = Path("output")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = output_dir / (Path(uploaded_file.name).stem + ".md")

                    # Process file
                    with st.spinner("Processing..."):
                        result, chunks_folder = convert_file(
                            Path(tmp_path),
                            output_path,
                            enable_chunking=enable_chunking,
                            chunk_strategy=chunk_strategy,
                            chunk_size=chunk_size,
                            overlap=overlap,
                            enable_embedding=enable_embeddings,
                            api_key=api_key,
                            postgres_url=postgres_url if enable_postgres else None
                        )

                        end_time = time.time()
                        processing_time = end_time - start_time

                    # Display results
                    st.success(f"✅ {result}")

                    # Show local save path
                    if chunks_folder and chunks_folder.exists():
                        st.info(f"📁 Saved to: `{chunks_folder.absolute()}`")
                    elif output_path.exists():
                        st.info(f"📁 Saved to: `{output_path.absolute()}`")

                    # Statistics
                    st.subheader("📊 Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)

                    with stats_col1:
                        st.metric("Processing Time", f"{processing_time:.2f}s")

                    with stats_col2:
                        st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")

                    # Read chunks if created
                    if chunks_folder and chunks_folder.exists():
                        chunk_files = sorted(chunks_folder.glob("chunk_*.md"))

                        with stats_col3:
                            st.metric("Chunks Created", len(chunk_files))

                        # Chunk preview
                        st.subheader("📝 Chunk Preview")

                        if chunk_files:
                            selected_chunk = st.selectbox(
                                "Select chunk to preview",
                                range(1, len(chunk_files) + 1),
                                format_func=lambda x: f"Chunk {x}"
                            )

                            chunk_content = chunk_files[selected_chunk - 1].read_text(encoding="utf-8")
                            st.markdown(chunk_content, unsafe_allow_html=True)

                        # Download buttons
                        st.subheader("⬇️ Downloads")

                        for chunk_file in chunk_files:
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.text(chunk_file.name)
                            with col_b:
                                st.download_button(
                                    "Download",
                                    chunk_file.read_bytes(),
                                    file_name=chunk_file.name,
                                    mime="text/markdown",
                                    key=f"dl_{chunk_file.name}"
                                )

                        # Embeddings download
                        embeddings_file = chunks_folder / "embeddings.json"
                        if embeddings_file.exists():
                            st.download_button(
                                "📥 Download embeddings.json",
                                embeddings_file.read_bytes(),
                                file_name="embeddings.json",
                                mime="application/json"
                            )

                    else:
                        # Single file output
                        if output_path.exists():
                            content = output_path.read_text(encoding="utf-8")

                            st.subheader("📝 Document Preview")
                            st.markdown(content[:2000] + "..." if len(content) > 2000 else content, unsafe_allow_html=True)

                            st.download_button(
                                "📥 Download Markdown",
                                content,
                                file_name=output_path.name,
                                mime="text/markdown"
                            )

                    # Database status
                    if enable_postgres and postgres_url:
                        st.subheader("💾 Database Status")
                        try:
                            with psycopg.connect(postgres_url) as conn:
                                with conn.cursor() as cur:
                                    cur.execute("SELECT COUNT(*) FROM document_chunks")
                                    total_chunks = cur.fetchone()[0]
                                    st.success(f"Total chunks in database: {total_chunks}")
                        except Exception as e:
                            st.error(f"Database error: {e}")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

# Tab 2: Batch upload
with tab2:
    st.header("Batch Upload Files")

    uploaded_files = st.file_uploader(
        "Choose multiple files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="batch"
    )

    if uploaded_files:
        st.info(f"📦 {len(uploaded_files)} file(s) selected")

        if st.button("🚀 Process All Files", key="process_batch"):
            start_time = time.time()

            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            success_count = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Create local output directory
                    output_dir = Path("output")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = output_dir / (Path(uploaded_file.name).stem + ".md")

                    # Process file
                    result, chunks_folder = convert_file(
                        Path(tmp_path),
                        output_path,
                        enable_chunking=enable_chunking,
                        chunk_strategy=chunk_strategy,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        enable_embedding=enable_embeddings,
                        api_key=api_key,
                        postgres_url=postgres_url if enable_postgres else None
                    )

                    # Show local path in result
                    local_path = chunks_folder.absolute() if chunks_folder else output_path.absolute()
                    results.append({
                        "filename": uploaded_file.name,
                        "status": "✅ Success",
                        "result": f"{result}\n📁 Saved to: {local_path}"
                    })
                    success_count += 1

                except Exception as e:
                    results.append({
                        "filename": uploaded_file.name,
                        "status": "❌ Failed",
                        "result": str(e)
                    })

                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))

            end_time = time.time()
            processing_time = end_time - start_time

            status_text.text("")
            progress_bar.empty()

            # Summary
            st.subheader("📊 Batch Processing Summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(uploaded_files))
            with col2:
                st.metric("Successful", success_count)
            with col3:
                st.metric("Processing Time", f"{processing_time:.2f}s")

            # Results table
            st.subheader("📋 Results")
            for result in results:
                with st.expander(f"{result['status']} {result['filename']}"):
                    st.text(result['result'])

            # Database status
            if enable_postgres and postgres_url:
                st.subheader("💾 Database Status")
                try:
                    with psycopg.connect(postgres_url) as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT COUNT(*) FROM document_chunks")
                            total_chunks = cur.fetchone()[0]
                            st.success(f"Total chunks in database: {total_chunks}")
                except Exception as e:
                    st.error(f"Database error: {e}")

# Tab 3: Database Browser
with tab3:
    st.header("💾 Database Browser")

    if not postgres_url:
        st.warning("⚠️ PostgreSQL not configured. Set POSTGRES_URL in .env file.")
    else:
        try:
            with psycopg.connect(postgres_url) as conn:
                with conn.cursor() as cur:
                    # Get total stats
                    cur.execute("SELECT COUNT(*) FROM document_chunks")
                    total_chunks = cur.fetchone()[0]

                    cur.execute("SELECT COUNT(DISTINCT filename) FROM document_chunks")
                    total_docs = cur.fetchone()[0]

                    # Display stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Documents", total_docs)
                    with col2:
                        st.metric("Total Chunks", total_chunks)

                    st.markdown("---")

                    # Get document list with stats
                    cur.execute("""
                        SELECT
                            filename,
                            COUNT(*) as chunk_count,
                            COUNT(embedding) as vector_count,
                            split_strategy,
                            MIN(created_at) as first_indexed,
                            MAX(created_at) as last_indexed
                        FROM document_chunks
                        GROUP BY filename, split_strategy
                        ORDER BY last_indexed DESC
                    """)
                    documents = cur.fetchall()

                    if documents:
                        st.subheader("📚 Indexed Documents")

                        # Create document selector
                        doc_options = [f"{doc[0]} ({doc[1]} chunks)" for doc in documents]
                        selected_doc_idx = st.selectbox(
                            "Select document to view",
                            range(len(doc_options)),
                            format_func=lambda x: doc_options[x]
                        )

                        if selected_doc_idx is not None:
                            selected_doc = documents[selected_doc_idx]
                            filename, chunk_count, vector_count, strategy, first_indexed, last_indexed = selected_doc

                            # Document details
                            st.subheader(f"📄 {filename}")

                            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                            with info_col1:
                                st.metric("Chunks", chunk_count)
                            with info_col2:
                                st.metric("Vectors", vector_count)
                            with info_col3:
                                st.metric("Strategy", strategy)
                            with info_col4:
                                st.metric("Indexed", first_indexed.strftime("%Y-%m-%d"))

                            # Search within document
                            st.markdown("---")
                            st.subheader("🔍 Search Chunks")

                            search_query = st.text_input("Search for text in chunks", key="search_chunks")

                            # Fetch chunks for this document
                            if search_query:
                                cur.execute("""
                                    SELECT chunk_text, embedding, created_at
                                    FROM document_chunks
                                    WHERE filename = %s AND chunk_text ILIKE %s
                                    ORDER BY created_at
                                    LIMIT 50
                                """, (filename, f"%{search_query}%"))
                            else:
                                cur.execute("""
                                    SELECT chunk_text, embedding, created_at
                                    FROM document_chunks
                                    WHERE filename = %s
                                    ORDER BY created_at
                                    LIMIT 50
                                """, (filename,))

                            chunks = cur.fetchall()

                            st.write(f"Showing {len(chunks)} chunk(s)")

                            # Display chunks
                            for idx, (chunk_text, embedding, created_at) in enumerate(chunks, 1):
                                vector_status = "✅ Has vector" if embedding else "❌ No vector"
                                vector_dim = f"({len(embedding)} dims)" if embedding else ""

                                with st.expander(f"Chunk {idx} - {created_at.strftime('%Y-%m-%d %H:%M')} - {vector_status} {vector_dim}"):
                                    st.markdown(f'<div dir="rtl">{chunk_text[:1000]}</div>', unsafe_allow_html=True)
                                    if len(chunk_text) > 1000:
                                        st.caption(f"... ({len(chunk_text) - 1000} more characters)")

                                    if embedding:
                                        st.markdown("---")
                                        st.caption(f"**Embedding Vector** ({len(embedding)} dimensions)")

                                        with st.expander("Show vector values"):
                                            # Display first 10 values as preview
                                            st.code(f"First 10 values: {embedding[:10]}")
                                            st.code(f"Last 10 values: {embedding[-10:]}")

                                            # Full vector download
                                            import json
                                            vector_json = json.dumps({"embedding": embedding, "dimension": len(embedding)}, indent=2)
                                            st.download_button(
                                                "Download full vector",
                                                vector_json,
                                                file_name=f"chunk_{idx}_vector.json",
                                                mime="application/json",
                                                key=f"vector_dl_{filename}_{idx}"
                                            )

                            # Delete document option
                            st.markdown("---")
                            st.subheader("🗑️ Delete Document")
                            if st.button(f"Delete '{filename}' from database", key=f"delete_{filename}"):
                                try:
                                    cur.execute("DELETE FROM document_chunks WHERE filename = %s", (filename,))
                                    conn.commit()
                                    st.success(f"✅ Deleted {filename} from database")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed to delete: {e}")
                    else:
                        st.info("No documents in database yet. Upload files to index them.")

        except Exception as e:
            st.error(f"❌ Database error: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**RAG Document Indexer**")
st.sidebar.markdown("Hebrew RTL support | Multiple chunking strategies | Gemini embeddings")
