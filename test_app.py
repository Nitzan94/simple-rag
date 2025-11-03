# ABOUTME: Tests for Streamlit app and core document processing functions
# ABOUTME: Validates file processing, chunking, and embeddings workflow

import pytest
import tempfile
import os
from pathlib import Path
from index_documents import (
    chunk_text_fixed_size,
    chunk_text_by_sentences,
    chunk_text_by_paragraphs,
    chunk_text,
    convert_file,
    is_hebrew,
    reverse_hebrew_text
)


def test_is_hebrew():
    """Test Hebrew character detection"""
    assert is_hebrew('א') == True
    assert is_hebrew('ת') == True
    assert is_hebrew('a') == False
    assert is_hebrew('1') == False


def test_reverse_hebrew_text():
    """Test Hebrew text reversal while preserving English"""
    text = "שלום World"
    result = reverse_hebrew_text(text)
    # Hebrew should be reversed, English should stay
    assert 'World' in result
    assert result != text


def test_chunk_text_fixed_size():
    """Test fixed-size chunking with overlap"""
    text = "a" * 1000 + "b" * 1000
    chunks = chunk_text_fixed_size(text, chunk_size=1000, overlap=200)

    assert len(chunks) > 1
    assert len(chunks[0]) == 1000

    # Check overlap
    assert chunks[0][-200:] == chunks[1][:200]


def test_chunk_text_fixed_size_invalid_params():
    """Test that invalid parameters raise error"""
    with pytest.raises(ValueError):
        chunk_text_fixed_size("test", chunk_size=100, overlap=200)


def test_chunk_text_by_sentences():
    """Test sentence-based chunking"""
    text = "First sentence. Second sentence! Third sentence?"
    chunks = chunk_text_by_sentences(text)

    assert len(chunks) == 3
    assert "First sentence." in chunks[0]
    assert "Second sentence!" in chunks[1]
    assert "Third sentence?" in chunks[2]


def test_chunk_text_by_paragraphs():
    """Test paragraph-based chunking"""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text_by_paragraphs(text)

    assert len(chunks) == 3
    assert "First paragraph" in chunks[0]
    assert "Second paragraph" in chunks[1]
    assert "Third paragraph" in chunks[2]


def test_chunk_text_strategy_selection():
    """Test that chunk_text selects correct strategy"""
    text = "a" * 2000

    # Fixed strategy
    fixed_chunks = chunk_text(text, strategy='fixed', chunk_size=1000, overlap=200)
    assert len(fixed_chunks) > 1

    # Sentence strategy
    sentence_text = "Sentence one. Sentence two. Sentence three."
    sentence_chunks = chunk_text(sentence_text, strategy='sentence')
    assert len(sentence_chunks) == 3

    # Paragraph strategy
    para_text = "Para one.\n\nPara two.\n\nPara three."
    para_chunks = chunk_text(para_text, strategy='paragraph')
    assert len(para_chunks) == 3


def test_chunk_text_invalid_strategy():
    """Test that invalid strategy raises error"""
    with pytest.raises(ValueError):
        chunk_text("test", strategy='invalid')


def test_convert_txt_file():
    """Test converting TXT file to markdown"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test TXT file
        txt_path = Path(tmpdir) / "test.txt"
        txt_path.write_text("Test content\nLine 2", encoding="utf-8")

        # Convert
        output_path = Path(tmpdir) / "test.md"
        result, chunks_folder = convert_file(
            txt_path,
            output_path,
            enable_chunking=False
        )

        # Verify
        assert output_path.exists()
        assert "Converted" in result
        content = output_path.read_text(encoding="utf-8")
        assert "Test content" in content
        assert 'dir="rtl"' in content


def test_convert_txt_file_with_chunking():
    """Test converting TXT file with chunking enabled"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test TXT file with enough content for multiple chunks
        txt_path = Path(tmpdir) / "test.txt"
        content = "a" * 1500  # Enough for 2 chunks with default settings
        txt_path.write_text(content, encoding="utf-8")

        # Convert with chunking
        output_path = Path(tmpdir) / "test.md"
        result, chunks_folder = convert_file(
            txt_path,
            output_path,
            enable_chunking=True,
            chunk_strategy='fixed',
            chunk_size=1000,
            overlap=200
        )

        # Verify
        assert chunks_folder is not None
        assert chunks_folder.exists()
        assert "chunks" in result

        # Check chunk files exist
        chunk_files = list(chunks_folder.glob("chunk_*.md"))
        assert len(chunk_files) > 0


def test_streamlit_app_import():
    """Test that app.py can be imported without errors"""
    try:
        import app
        assert True
    except Exception as e:
        pytest.fail(f"Failed to import app.py: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
