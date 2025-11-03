# Simple RAG - Document to Markdown Converter

Document to Markdown converter with Hebrew RTL support, multiple chunking strategies, and Google Gemini embeddings.

## Features

- **File Formats**: PDF, DOCX, TXT
- **Hebrew Support**: Automatic text reversal for PDFs, RTL formatting
- **Chunking Strategies**:
  - Fixed-size with overlap
  - Sentence-based splitting
  - Paragraph-based splitting
- **Embeddings**: Google Gemini API integration
- **Storage**: PostgreSQL with pgvector for vector similarity search
- **Output**: Organized chunk folders with embeddings.json + optional DB storage

## Installation

```bash
# Clone or navigate to project
cd simple-rag

# Install with uv (recommended)
uv sync
```

## Setup

Create `.env` file:

```env
GEMINI_API_KEY=your_api_key_here

# Optional - PostgreSQL for vector storage
POSTGRES_URL=postgresql://postgres:mypassword@localhost:5432/simple_rag
```

### PostgreSQL (Optional)

Docker setup:

```bash
docker pull pgvector/pgvector:pg16
docker run -d --name pgvector -e POSTGRES_PASSWORD=mypassword -p 5432:5432 pgvector/pgvector:pg16
docker exec pgvector psql -U postgres -c "CREATE DATABASE simple_rag;"
```

Table auto-created: id (UUID), chunk_text, embedding (vector 768), filename, split_strategy, created_at

## Usage

### Streamlit UI (Recommended)

Run the web interface:

```bash
uv run streamlit run app.py
```

Features:
- **Single File / Batch Upload**: Upload one or multiple files via tabs
- **Live Settings**: Configure chunking strategy, embeddings, PostgreSQL in sidebar
- **Real-time Preview**: View chunks inline with markdown rendering
- **Downloads**: Download individual chunks and embeddings.json
- **Statistics**: Processing time, file size, chunk count
- **Database Status**: View PostgreSQL connection and total chunks stored

### CLI (Command Line)

Run the converter:

```bash
uv run python index_documents.py
```

#### Interactive Workflow

1. **Choose conversion mode**:
   - `1` - Convert a specific file
   - `2` - Convert all files in a directory

2. **Enable chunking?** (y/n)
   - If yes, choose strategy:
     - `1` - Fixed-size with overlap (default: 1000 chars, 200 overlap)
     - `2` - Sentence-based splitting
     - `3` - Paragraph-based splitting

3. **Generate embeddings?** (y/n)
   - Uses Google Gemini API
   - Requires GEMINI_API_KEY in .env

### Output Structure

Without chunking:
```
output/
 document.md
```

With chunking:
```
output/
 document/
     chunk_1.md
     chunk_2.md
     chunk_3.md
     embeddings.json  (if enabled)
```

### Example embeddings.json

```json
[
  {
    "chunk_id": 1,
    "chunk_file": "chunk_1.md",
    "text": "Preview of chunk text...",
    "embedding": [0.123, -0.456, ...],
    "embedding_dim": 768
  }
]
```

## Project Structure

```
simple-rag/
 .env                      # API keys (not in git)
 .venv/                    # Virtual environment
 app.py                    # Streamlit UI
 index_documents.py        # Core converter functions (CLI + lib)
 pyproject.toml            # Project dependencies
 README.md                 # This file
 output/                   # Converted files
```

## Dependencies

- `pdfplumber>=0.11.7` - PDF text extraction
- `python-docx>=1.2.0` - DOCX file processing
- `python-dotenv>=1.2.1` - Environment variable loading
- `google-generativeai>=0.8.5` - Gemini embeddings
- `psycopg[binary]>=3.1.0` - PostgreSQL driver
- `pgvector>=0.4.1` - Vector extension for PostgreSQL
- `streamlit>=1.51.0` - Web UI framework

## Development

Add new dependencies:

```bash
uv add package-name
```

Run in development:

```bash
uv run python index_documents.py
```

## License

MIT
