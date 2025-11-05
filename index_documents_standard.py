# ABOUTME: Document indexing RAG pipeline with PostgreSQL storage
# ABOUTME: Provides document conversion, chunking, embeddings, and database storage

import pdfplumber
import re
import os
import json
from pathlib import Path
from docx import Document
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

def is_hebrew(char):
    """Check if character is Hebrew"""
    return '\u0590' <= char <= '\u05FF'

def reverse_hebrew_text(text):
    """Reverse Hebrew text while preserving English, numbers, and punctuation order"""
    tokens = re.findall(r'\S+|\s+', text)

    result = []
    for token in tokens:
        if token.strip():
            if any(is_hebrew(c) for c in token):
                result.append(token[::-1])
            else:
                result.append(token)
        else:
            result.append(token)

    return ''.join(result)

def chunk_text_fixed_size(text, chunk_size=1000, overlap=200):
    """
    Split text into fixed-size chunks with overlap

    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks

def chunk_text_by_sentences(text):
    """
    Split text into chunks by sentences

    Args:
        text: The text to chunk

    Returns:
        List of text chunks (one or more sentences per chunk)
    """
    import re

    # Split by common sentence endings (., !, ?, newlines)
    # Keep the delimiter with the sentence
    sentences = re.split(r'(?<=[.!?\n])\s+', text)

    # Filter out empty strings
    chunks = [s.strip() for s in sentences if s.strip()]

    return chunks

def chunk_text_by_paragraphs(text):
    """
    Split text into chunks by paragraphs

    Args:
        text: The text to chunk

    Returns:
        List of text chunks (paragraphs)
    """
    # Split by double newlines (paragraph breaks)
    paragraphs = text.split('\n\n')

    # Filter out empty strings
    chunks = [p.strip() for p in paragraphs if p.strip()]

    return chunks

def chunk_text(text, strategy='fixed', chunk_size=1000, overlap=200):
    """
    Split text into chunks using specified strategy

    Args:
        text: The text to chunk
        strategy: 'fixed', 'sentence', or 'paragraph'
        chunk_size: Size of each chunk in characters (for fixed strategy)
        overlap: Number of overlapping characters (for fixed strategy)

    Returns:
        List of text chunks
    """
    if strategy == 'fixed':
        return chunk_text_fixed_size(text, chunk_size, overlap)
    elif strategy == 'sentence':
        return chunk_text_by_sentences(text)
    elif strategy == 'paragraph':
        return chunk_text_by_paragraphs(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

def format_text_to_markdown(text):
    """Format text content to markdown with RTL Hebrew support (no reversal needed for DOCX/TXT)"""
    markdown_content = []
    markdown_content.append('<div dir="rtl">\n\n')

    lines = text.split('\n')
    for line in lines:
        if line.strip():
            markdown_content.append(f"{line.strip()}\n")
        else:
            markdown_content.append("\n")

    markdown_content.append('\n</div>')
    return "".join(markdown_content)

def convert_pdf_to_markdown(pdf_path, output_path):
    """Convert single PDF to markdown with RTL Hebrew support (reverses Hebrew from PDF)"""
    markdown_content = []
    markdown_content.append('<div dir="rtl">\n\n')

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()

            if text:
                markdown_content.append(f"# עמוד {page_num}\n\n")
                lines = text.split('\n')
                for line in lines:
                    if line.strip():
                        fixed_line = reverse_hebrew_text(line)
                        markdown_content.append(f"{fixed_line}\n")
                    else:
                        markdown_content.append("\n")
                markdown_content.append("\n---\n\n")

    markdown_content.append('</div>')
    full_markdown = "".join(markdown_content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    return len(pdf.pages)

def convert_txt_to_markdown(txt_path, output_path):
    """Convert TXT file to markdown with RTL Hebrew support"""
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    markdown = format_text_to_markdown(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return 1  # Single "page"

def convert_docx_to_markdown(docx_path, output_path):
    """Convert DOCX file to markdown with RTL Hebrew support"""
    doc = Document(docx_path)

    # Extract all text from paragraphs
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)

    text = "\n".join(text_parts)
    markdown = format_text_to_markdown(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return 1  # Single "page"

def generate_embedding(text, api_key):
    """Generate embedding for text using Google Gemini API"""
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"[WARN] Failed to generate embedding: {e}")
        return None

def create_table(conn):
    """Create document_chunks table for standard PostgreSQL (no pgvector extension)"""
    try:
        with conn.cursor() as cur:
            # Create table with FLOAT[] for embeddings (standard PostgreSQL)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chunk_text TEXT NOT NULL,
                    embedding FLOAT[],
                    filename TEXT NOT NULL,
                    split_strategy TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create standard index for filename searches
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_filename
                ON document_chunks (filename)
            """)

            conn.commit()
            print("  [DATABASE] Table created/verified successfully")
    except Exception as e:
        print(f"  [WARN] Failed to create table: {e}")
        conn.rollback()

def save_to_database(chunks_data, postgres_url):
    """Save chunks and embeddings to standard PostgreSQL database"""
    if not postgres_url:
        return False

    try:
        with psycopg.connect(postgres_url) as conn:
            # Create table if doesn't exist
            create_table(conn)

            with conn.cursor() as cur:
                for chunk_data in chunks_data:
                    cur.execute("""
                        INSERT INTO document_chunks
                        (chunk_text, embedding, filename, split_strategy, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        chunk_data['chunk_text'],
                        chunk_data['embedding'],
                        chunk_data['filename'],
                        chunk_data['split_strategy'],
                        chunk_data.get('created_at', datetime.now())
                    ))

            conn.commit()
            print(f"  [DATABASE] Saved {len(chunks_data)} chunks to PostgreSQL")
            return True

    except Exception as e:
        print(f"  [WARN] Failed to save to database: {e}")
        return False

def save_chunks(chunks, base_output_path, enable_embedding=False, api_key=None, split_strategy='fixed', postgres_url=None):
    """Save chunks as separate numbered files in a dedicated folder"""
    saved_files = []
    base_name = base_output_path.stem
    output_parent_dir = base_output_path.parent

    # Create folder for chunks
    chunks_folder = output_parent_dir / base_name
    os.makedirs(chunks_folder, exist_ok=True)

    embeddings_data = []

    for i, chunk in enumerate(chunks, 1):
        chunk_filename = f"chunk_{i}.md"
        chunk_path = chunks_folder / chunk_filename

        # Save markdown chunk
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(f'<div dir="rtl">\n\n')
            f.write(f"# Chunk {i}/{len(chunks)}\n\n")
            f.write(chunk)
            f.write('\n\n</div>')

        saved_files.append(chunk_filename)

        # Generate embedding if enabled
        if enable_embedding and api_key:
            print(f"  [EMBEDDING] Generating embedding for chunk {i}/{len(chunks)}...")
            embedding = generate_embedding(chunk, api_key)
            if embedding:
                embeddings_data.append({
                    "chunk_id": i,
                    "chunk_file": chunk_filename,
                    "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,  # Preview
                    "embedding": embedding,
                    "embedding_dim": len(embedding)
                })

    # Save embeddings to JSON file
    if embeddings_data:
        embeddings_file = chunks_folder / "embeddings.json"
        with open(embeddings_file, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        print(f"  [EMBEDDING] Saved {len(embeddings_data)} embeddings to embeddings.json")

        # Save to database if configured
        if postgres_url:
            # Prepare database data
            db_chunks_data = []
            for chunk_data in embeddings_data:
                db_chunks_data.append({
                    'chunk_text': chunks[chunk_data['chunk_id'] - 1],
                    'embedding': chunk_data['embedding'],
                    'filename': base_name,
                    'split_strategy': split_strategy,
                    'created_at': datetime.now()
                })

            # Save to database
            save_to_database(db_chunks_data, postgres_url)

    return chunks_folder, saved_files

def convert_file(file_path, output_path, enable_chunking=False, chunk_strategy='fixed',
                 chunk_size=1000, overlap=200, enable_embedding=False, api_key=None, postgres_url=None):
    """Convert a single file based on its extension"""
    if file_path.suffix.lower() == '.pdf':
        pages = convert_pdf_to_markdown(str(file_path), str(output_path))
        result = f"Converted {pages} pages"
    elif file_path.suffix.lower() == '.txt':
        convert_txt_to_markdown(str(file_path), str(output_path))
        result = "Converted"
    elif file_path.suffix.lower() == '.docx':
        convert_docx_to_markdown(str(file_path), str(output_path))
        result = "Converted"
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # Apply chunking if enabled
    chunks_folder = None
    if enable_chunking:
        # Read the generated markdown
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove RTL wrapper for chunking
        content = content.replace('<div dir="rtl">', '').replace('</div>', '').strip()

        # Create chunks using selected strategy
        chunks = chunk_text(content, chunk_strategy, chunk_size, overlap)

        # Save chunks to dedicated folder (with optional embeddings and database)
        chunks_folder, saved_files = save_chunks(chunks, Path(output_path), enable_embedding, api_key,
                                                 chunk_strategy, postgres_url)

        # Remove original file
        os.remove(output_path)

        strategy_name = {'fixed': 'fixed-size', 'sentence': 'sentence-based', 'paragraph': 'paragraph-based'}
        result += f" -> {len(chunks)} chunks ({strategy_name[chunk_strategy]}) in folder '{chunks_folder.name}'"
        if enable_embedding:
            result += " (with embeddings)"

    return result, chunks_folder

def process_single_file():
    """Process a single file"""
    file_path = input("\nEnter file path: ").strip().strip('"').strip("'")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    if not os.path.isfile(file_path):
        print(f"[ERROR] Path is not a file: {file_path}")
        return

    file_path = Path(file_path)

    # Check if supported
    if file_path.suffix.lower() not in ['.pdf', '.txt', '.docx']:
        print(f"[ERROR] Unsupported file type: {file_path.suffix}")
        print("Supported types: .pdf, .txt, .docx")
        return

    # Get output directory (optional)
    output_dir = input("Enter output directory (press Enter for same directory as file): ").strip()
    output_dir = output_dir.strip('"').strip("'")

    if not output_dir:
        output_dir = file_path.parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Ask about chunking
    enable_chunking = input("\nEnable chunking? (y/n, default: n): ").strip().lower() == 'y'
    chunk_strategy = 'fixed'
    chunk_size = 1000
    overlap = 200
    enable_embedding = False
    api_key = None

    if enable_chunking:
        # Ask for chunking strategy
        print("\nChoose chunking strategy:")
        print("1. Fixed-size with overlap")
        print("2. Sentence-based splitting")
        print("3. Paragraph-based splitting")
        strategy_choice = input("Enter choice (1, 2, or 3, default: 1): ").strip()

        if strategy_choice == '2':
            chunk_strategy = 'sentence'
        elif strategy_choice == '3':
            chunk_strategy = 'paragraph'
        else:
            chunk_strategy = 'fixed'

        # Only ask for size/overlap if using fixed strategy
        if chunk_strategy == 'fixed':
            chunk_input = input("Chunk size in characters (default: 1000): ").strip()
            if chunk_input:
                chunk_size = int(chunk_input)

            overlap_input = input("Overlap size in characters (default: 200): ").strip()
            if overlap_input:
                overlap = int(overlap_input)

        # Ask about embeddings
        enable_embedding = input("\nGenerate embeddings with Gemini API? (y/n, default: n): ").strip().lower() == 'y'
        if enable_embedding:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("[WARN] GEMINI_API_KEY not found in .env file. Embeddings disabled.")
                enable_embedding = False

    # Get PostgreSQL URL from .env (optional)
    postgres_url = os.getenv("POSTGRES_URL")

    output_filename = file_path.stem + ".md"
    output_path = output_dir / output_filename

    try:
        print(f"\n[PROCESSING] {file_path.name}")
        result, chunks_folder = convert_file(file_path, output_path, enable_chunking, chunk_strategy,
                                            chunk_size, overlap, enable_embedding, api_key, postgres_url)
        print(f"[OK] {result}")
        if enable_chunking and chunks_folder:
            print(f"[OUTPUT] Chunks saved to: {chunks_folder}")
        else:
            print(f"[OUTPUT] Saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to convert: {e}")

def process_directory():
    """Process all files in a directory"""
    input_dir = input("\nEnter directory path containing documents: ").strip()
    input_dir = input_dir.strip('"').strip("'")

    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        return

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Path is not a directory: {input_dir}")
        return

    # Get output directory (optional)
    output_dir = input("Enter output directory (press Enter for same directory): ").strip()
    output_dir = output_dir.strip('"').strip("'")

    if not output_dir:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Ask about chunking
    enable_chunking = input("\nEnable chunking? (y/n, default: n): ").strip().lower() == 'y'
    chunk_strategy = 'fixed'
    chunk_size = 1000
    overlap = 200
    enable_embedding = False
    api_key = None

    if enable_chunking:
        # Ask for chunking strategy
        print("\nChoose chunking strategy:")
        print("1. Fixed-size with overlap")
        print("2. Sentence-based splitting")
        print("3. Paragraph-based splitting")
        strategy_choice = input("Enter choice (1, 2, or 3, default: 1): ").strip()

        if strategy_choice == '2':
            chunk_strategy = 'sentence'
        elif strategy_choice == '3':
            chunk_strategy = 'paragraph'
        else:
            chunk_strategy = 'fixed'

        # Only ask for size/overlap if using fixed strategy
        if chunk_strategy == 'fixed':
            chunk_input = input("Chunk size in characters (default: 1000): ").strip()
            if chunk_input:
                chunk_size = int(chunk_input)

            overlap_input = input("Overlap size in characters (default: 200): ").strip()
            if overlap_input:
                overlap = int(overlap_input)

        # Ask about embeddings
        enable_embedding = input("\nGenerate embeddings with Gemini API? (y/n, default: n): ").strip().lower() == 'y'
        if enable_embedding:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("[WARN] GEMINI_API_KEY not found in .env file. Embeddings disabled.")
                enable_embedding = False

    # Get PostgreSQL URL from .env (optional)
    postgres_url = os.getenv("POSTGRES_URL")

    # Find all supported files
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob("*.pdf"))
    txt_files = list(input_path.glob("*.txt"))
    docx_files = list(input_path.glob("*.docx"))

    all_files = pdf_files + txt_files + docx_files

    if not all_files:
        print(f"\n[WARN] No supported files (PDF, TXT, DOCX) found in: {input_dir}")
        return

    print(f"\n[INFO] Found {len(all_files)} file(s):")
    print(f"  - PDF: {len(pdf_files)}")
    print(f"  - TXT: {len(txt_files)}")
    print(f"  - DOCX: {len(docx_files)}")
    if enable_chunking:
        strategy_names = {'fixed': 'Fixed-size', 'sentence': 'Sentence-based', 'paragraph': 'Paragraph-based'}
        print(f"\n[CHUNKING] Strategy: {strategy_names[chunk_strategy]}")
        if chunk_strategy == 'fixed':
            print(f"[CHUNKING] Size: {chunk_size} chars, Overlap: {overlap} chars")
        if enable_embedding:
            print(f"[EMBEDDING] Enabled with Gemini API")
    print("-" * 50)

    # Convert each file
    success_count = 0
    chunks_folders = []
    for file_path in all_files:
        try:
            output_filename = file_path.stem + ".md"
            output_path = Path(output_dir) / output_filename

            print(f"\n[PROCESSING] {file_path.name}")
            result, chunks_folder = convert_file(file_path, output_path, enable_chunking, chunk_strategy,
                                                chunk_size, overlap, enable_embedding, api_key, postgres_url)
            print(f"[OK] {result}")
            if chunks_folder:
                chunks_folders.append(chunks_folder)
            success_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to convert {file_path.name}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print(f"[SUMMARY] Successfully converted {success_count}/{len(all_files)} files")
    if enable_chunking and chunks_folders:
        print(f"[OUTPUT] {len(chunks_folders)} chunk folders created in: {output_dir}")
    else:
        print(f"[OUTPUT] Files saved to: {output_dir}")

def main():
    print("Document to Markdown Converter")
    print("Supports: PDF, DOCX, TXT")
    print("Features: Hebrew text reversal (PDF), RTL formatting")
    print("Chunking: Fixed-size, Sentence-based, Paragraph-based")
    print("Embeddings: Google Gemini API")
    print("Database: PostgreSQL with vector storage")
    print("=" * 70)

    # Ask for mode
    print("\nChoose conversion mode:")
    print("1. Convert a specific file")
    print("2. Convert all files in a directory")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        process_single_file()
    elif choice == '2':
        process_directory()
    else:
        print("[ERROR] Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
