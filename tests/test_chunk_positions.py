"""
Unit tests for chunk position tracking.

Tests that character positions (start_char, end_char) are correctly tracked
through the chunking pipeline, enabling snippet-level evaluation.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retrieval.chunking import RecursiveCharacterTextSplitter, DocumentChunker


def test_basic_position_tracking():
    """Test that positions are tracked correctly for simple text."""
    text = "A" * 200 + "B" * 200 + "C" * 200
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    chunks = splitter.split_text_with_positions(text)

    # Verify all chunks have positions
    for chunk_text, start, end in chunks:
        assert start >= 0, f"Start position should be non-negative: {start}"
        assert end > start, f"End should be after start: start={start}, end={end}"
        # Verify the text at these positions matches the chunk
        assert text[start:end] == chunk_text, \
            f"Text mismatch: text[{start}:{end}] != chunk_text"

    print(f"✓ Basic position tracking: {len(chunks)} chunks verified")


def test_overlapping_chunks():
    """Test position tracking with overlapping chunks."""
    text = "ABCDEF" * 50  # 300 chars
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    chunks = splitter.split_text_with_positions(text)

    # Check that overlaps exist
    overlaps_found = 0
    for i in range(len(chunks) - 1):
        chunk1_text, start1, end1 = chunks[i]
        chunk2_text, start2, end2 = chunks[i+1]

        # Next chunk should start before previous one ends (overlap)
        if start2 < end1:
            overlaps_found += 1
            overlap_size = end1 - start2
            assert overlap_size > 0, f"Overlap size should be positive: {overlap_size}"

    assert overlaps_found > 0, "Should have at least some overlapping chunks"
    print(f"✓ Overlapping chunks: {overlaps_found} overlaps found in {len(chunks)} chunks")


def test_document_chunker_positions():
    """Test DocumentChunker preserves positions."""
    text = "Legal document text here. " * 100  # ~2600 chars

    chunker = DocumentChunker({'chunk_size': 100, 'chunk_overlap': 20})
    chunks = chunker.chunk_document(text, metadata={'source': 'test.txt'})

    # All chunks should have position fields
    for chunk in chunks:
        assert 'start_char' in chunk, "Chunk should have 'start_char' field"
        assert 'end_char' in chunk, "Chunk should have 'end_char' field"
        assert chunk['start_char'] >= 0, f"Start char should be non-negative: {chunk['start_char']}"
        assert chunk['end_char'] > chunk['start_char'], \
            f"End should be after start: {chunk['start_char']} >= {chunk['end_char']}"

        # Verify position accuracy
        extracted = text[chunk['start_char']:chunk['end_char']]
        assert extracted == chunk['text'], \
            f"Position mismatch in chunk {chunk['chunk_id']}"

    print(f"✓ DocumentChunker positions: {len(chunks)} chunks verified")


def test_multi_document_chunking():
    """Test that positions are per-document, not global."""
    docs = [
        {'text': 'A' * 200, 'source': 'doc1.txt'},
        {'text': 'B' * 200, 'source': 'doc2.txt'},
    ]

    chunker = DocumentChunker({'chunk_size': 50, 'chunk_overlap': 10})
    chunks = chunker.chunk_documents(docs)

    # Group by source
    doc1_chunks = [c for c in chunks if c['source'] == 'doc1.txt']
    doc2_chunks = [c for c in chunks if c['source'] == 'doc2.txt']

    # First chunk of each doc should start at 0 (per-document coordinates)
    assert doc1_chunks[0]['start_char'] == 0, \
        f"First chunk of doc1 should start at 0, got {doc1_chunks[0]['start_char']}"
    assert doc2_chunks[0]['start_char'] == 0, \
        f"First chunk of doc2 should start at 0, got {doc2_chunks[0]['start_char']}"

    # Verify positions are within document bounds
    for chunk in doc1_chunks:
        assert chunk['end_char'] <= 200, \
            f"Doc1 chunk position exceeds document length: {chunk['end_char']}"

    for chunk in doc2_chunks:
        assert chunk['end_char'] <= 200, \
            f"Doc2 chunk position exceeds document length: {chunk['end_char']}"

    print(f"✓ Multi-document chunking: {len(doc1_chunks)} + {len(doc2_chunks)} chunks verified")


def test_backward_compatibility():
    """Test that old code without position tracking still works."""
    text = "Test text for backward compatibility. " * 20

    chunker = DocumentChunker({'chunk_size': 100, 'chunk_overlap': 20})

    # Test with track_positions=False
    chunks_no_pos = chunker.chunk_document(text, metadata={'source': 'test.txt'},
                                           track_positions=False)

    # Should still have the fields, but with default values
    for chunk in chunks_no_pos:
        assert 'start_char' in chunk
        assert 'end_char' in chunk
        # With track_positions=False, uses fallback (0, len(chunk_text))
        assert chunk['start_char'] == 0
        assert chunk['end_char'] == len(chunk['text'])

    print(f"✓ Backward compatibility: {len(chunks_no_pos)} chunks with legacy mode")


def test_realistic_legal_text():
    """Test with realistic legal document text."""
    legal_text = """
    MUTUAL NON-DISCLOSURE AGREEMENT

    This Non-Disclosure Agreement (the "Agreement") is entered into as of [Date]
    by and between [Party A] and [Party B] (collectively, the "Parties").

    1. DEFINITIONS

    1.1 "Confidential Information" means any information disclosed by one Party
    to the other Party, either directly or indirectly, in writing, orally, or
    by inspection of tangible objects.

    1.2 "Receiving Party" means the Party receiving Confidential Information.

    1.3 "Disclosing Party" means the Party disclosing Confidential Information.

    2. OBLIGATIONS OF RECEIVING PARTY

    2.1 The Receiving Party agrees to hold the Confidential Information in
    strict confidence and to take all reasonable precautions to protect such
    Confidential Information.

    2.2 The Receiving Party shall not, without prior written approval of the
    Disclosing Party, use for its own benefit, publish, copy, or otherwise
    disclose to others, or permit the use by others for their benefit or to
    the detriment of the Disclosing Party, any Confidential Information.
    """

    chunker = DocumentChunker({'chunk_size': 256, 'chunk_overlap': 50})
    chunks = chunker.chunk_document(legal_text,
                                    metadata={'source': 'nda.txt', 'type': 'contract'})

    # Verify all chunks
    for chunk in chunks:
        # Check position fields exist
        assert 'start_char' in chunk and 'end_char' in chunk

        # Verify extraction
        extracted = legal_text[chunk['start_char']:chunk['end_char']]
        assert extracted == chunk['text'], \
            f"Mismatch in chunk {chunk['chunk_id']}"

        # Check metadata preservation
        assert chunk['source'] == 'nda.txt'
        assert chunk['type'] == 'contract'

    print(f"✓ Realistic legal text: {len(chunks)} chunks from contract")


def test_edge_case_empty_text():
    """Test edge case: empty text."""
    chunker = DocumentChunker({'chunk_size': 100, 'chunk_overlap': 20})
    chunks = chunker.chunk_document('', metadata={'source': 'empty.txt'})

    # Should return empty list or single empty chunk
    assert len(chunks) <= 1, f"Empty text should produce at most 1 chunk, got {len(chunks)}"
    print(f"✓ Edge case (empty text): {len(chunks)} chunks")


def test_edge_case_very_short_text():
    """Test edge case: text shorter than chunk_size."""
    text = "Short text."
    chunker = DocumentChunker({'chunk_size': 1000, 'chunk_overlap': 20})
    chunks = chunker.chunk_document(text, metadata={'source': 'short.txt'})

    # Should be a single chunk
    assert len(chunks) == 1, f"Short text should produce 1 chunk, got {len(chunks)}"
    assert chunks[0]['start_char'] == 0
    assert chunks[0]['end_char'] == len(text)
    assert chunks[0]['text'] == text

    print(f"✓ Edge case (short text): 1 chunk verified")


def test_snippet_overlap_simulation():
    """Simulate snippet overlap calculation as used in evaluation."""
    # Create a document and chunk it
    text = "A" * 100 + "B" * 100 + "C" * 100 + "D" * 100  # 400 chars

    # Use smaller chunks with more overlap to ensure good snippet coverage
    chunker = DocumentChunker({'chunk_size': 120, 'chunk_overlap': 40})
    chunks = chunker.chunk_document(text, metadata={'source': 'test.txt'})

    # Simulate a ground truth snippet
    gt_snippet = {
        'file_path': 'test.txt',
        'span': (90, 150)  # 60 chars spanning A/B boundary
    }

    # Find which chunks overlap with this snippet
    overlapping_chunks = []
    for chunk in chunks:
        chunk_span = (chunk['start_char'], chunk['end_char'])
        gt_span = tuple(gt_snippet['span'])

        # Calculate IoU (Intersection over Union)
        intersection_start = max(chunk_span[0], gt_span[0])
        intersection_end = min(chunk_span[1], gt_span[1])

        if intersection_start < intersection_end:
            intersection = intersection_end - intersection_start
            union = max(chunk_span[1], gt_span[1]) - min(chunk_span[0], gt_span[0])
            iou = intersection / union

            if iou >= 0.5:  # Threshold used in LegalBench
                overlapping_chunks.append((chunk['chunk_id'], iou))

    # Should find at least one overlapping chunk
    assert len(overlapping_chunks) > 0, \
        f"Should find at least one chunk overlapping with ground truth snippet"

    print(f"✓ Snippet overlap simulation: {len(overlapping_chunks)} chunks overlap with GT")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_basic_position_tracking,
        test_overlapping_chunks,
        test_document_chunker_positions,
        test_multi_document_chunking,
        test_backward_compatibility,
        test_realistic_legal_text,
        test_edge_case_empty_text,
        test_edge_case_very_short_text,
        test_snippet_overlap_simulation,
    ]

    print("=" * 80)
    print("Running Chunk Position Tracking Tests")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1

    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
