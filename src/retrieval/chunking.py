"""
Document Chunking Module

Implements Recursive Character Text Splitter (RCTS) for legal documents.
Based on LegalBench-RAG findings that RCTS outperforms naive fixed-size chunking.
"""

from typing import List, Dict, Any, Optional
import re


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using a hierarchy of separators.

    Preserves document structure by maintaining paragraphs, sentences, and words together.
    This approach is particularly effective for legal documents which have structured formatting.

    Based on LangChain's RecursiveCharacterTextSplitter with optimizations for legal text.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators in order of preference (hierarchical)
            keep_separator: Whether to keep the separator in the chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator

        # Default hierarchical separators optimized for legal documents
        if separators is None:
            self.separators = [
                "\n\n",      # Paragraph breaks (highest priority)
                "\n",        # Line breaks
                ". ",        # Sentence endings
                "! ",        # Exclamation sentences
                "? ",        # Question sentences
                "; ",        # Semicolon breaks
                ", ",        # Comma breaks
                " ",         # Word breaks
                "",          # Character-level (last resort)
            ]
        else:
            self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive strategy.

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        return self._split_text_recursive(text, self.separators)

    def split_text_with_positions(self, text: str) -> List[tuple]:
        """
        Split text into chunks with character positions tracked.

        Args:
            text: Input text to split

        Returns:
            List of (chunk_text, start_char, end_char) tuples where positions
            are relative to the original text
        """
        # Use modified recursive splitting that tracks positions
        return self._split_text_with_positions_recursive(text, self.separators, offset=0)

    def _split_text_with_positions_recursive(
        self,
        text: str,
        separators: List[str],
        offset: int = 0
    ) -> List[tuple]:
        """
        Recursively split text using hierarchical separators while tracking positions.

        Args:
            text: Text to split
            separators: List of separators to try
            offset: Character offset from start of original document

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        final_chunks = []

        # Get the separator to use
        separator = separators[-1] if separators else ""
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by the separator and track positions
        splits_with_pos = self._split_by_separator_with_positions(text, separator, offset)

        # Merge splits into chunks
        good_splits = []
        current_pos = offset

        for split_text, split_start, split_end in splits_with_pos:
            if len(split_text) < self.chunk_size:
                good_splits.append((split_text, split_start, split_end))
            else:
                # Split is too large, process accumulated splits first
                if good_splits:
                    merged = self._merge_splits_with_positions(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    # No more separators, force split by character
                    # Just take first chunk_size characters
                    chunk_text = split_text[:self.chunk_size]
                    final_chunks.append((chunk_text, split_start, split_start + len(chunk_text)))
                else:
                    # Recursively split with next separator
                    recursive_chunks = self._split_text_with_positions_recursive(
                        split_text, new_separators, offset=split_start
                    )
                    final_chunks.extend(recursive_chunks)

        # Merge remaining splits
        if good_splits:
            merged = self._merge_splits_with_positions(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_separator_with_positions(
        self,
        text: str,
        separator: str,
        offset: int
    ) -> List[tuple]:
        """
        Split text by separator and track positions.

        Args:
            text: Text to split
            separator: Separator to use
            offset: Starting character position

        Returns:
            List of (split_text, start_pos, end_pos) tuples
        """
        if separator == "":
            # Character-level split
            return [(char, offset + i, offset + i + 1) for i, char in enumerate(text)]

        splits_with_pos = []

        if self.keep_separator:
            # Keep separator at the end of each split
            pattern = f"({re.escape(separator)})"
            parts = re.split(pattern, text)

            current_offset = offset
            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    split_text = parts[i] + parts[i + 1]
                else:
                    split_text = parts[i]

                if split_text:
                    start_pos = current_offset
                    end_pos = current_offset + len(split_text)
                    splits_with_pos.append((split_text, start_pos, end_pos))
                    current_offset = end_pos

            # Add the last part if exists
            if len(parts) % 2 == 1 and parts[-1]:
                split_text = parts[-1]
                start_pos = current_offset
                end_pos = current_offset + len(split_text)
                splits_with_pos.append((split_text, start_pos, end_pos))
        else:
            # Don't keep separator
            parts = text.split(separator)
            current_offset = offset
            for part in parts:
                if part:
                    start_pos = current_offset
                    end_pos = current_offset + len(part)
                    splits_with_pos.append((part, start_pos, end_pos))
                current_offset = end_pos + len(separator)

        return splits_with_pos

    def _merge_splits_with_positions(
        self,
        splits_with_pos: List[tuple],
        separator: str
    ) -> List[tuple]:
        """
        Merge small splits into chunks with overlap, preserving positions.

        Args:
            splits_with_pos: List of (text, start, end) tuples
            separator: Separator used for splitting

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        chunks = []
        current_splits = []
        current_length = 0

        for split_text, split_start, split_end in splits_with_pos:
            split_length = len(split_text)

            # Check if adding this split would exceed chunk_size
            if current_length + split_length + len(separator) > self.chunk_size:
                if current_splits:
                    # Save current chunk
                    chunk_text = separator.join([s[0] for s in current_splits]) if separator != "" else "".join([s[0] for s in current_splits])
                    chunk_start = current_splits[0][1]  # Start of first split
                    chunk_end = current_splits[-1][2]   # End of last split
                    chunks.append((chunk_text, chunk_start, chunk_end))

                    # Start new chunk with overlap
                    overlap_length = 0
                    overlap_splits = []
                    for prev_split in reversed(current_splits):
                        if overlap_length + len(prev_split[0]) < self.chunk_overlap:
                            overlap_splits.insert(0, prev_split)
                            overlap_length += len(prev_split[0]) + len(separator)
                        else:
                            break

                    current_splits = overlap_splits + [(split_text, split_start, split_end)]
                    current_length = sum(len(s[0]) for s in current_splits) + len(separator) * (len(current_splits) - 1)
                else:
                    # Single split is larger than chunk_size
                    current_splits = [(split_text, split_start, split_end)]
                    current_length = split_length
            else:
                current_splits.append((split_text, split_start, split_end))
                current_length += split_length + (len(separator) if current_splits else 0)

        # Add the last chunk
        if current_splits:
            chunk_text = separator.join([s[0] for s in current_splits]) if separator != "" else "".join([s[0] for s in current_splits])
            chunk_start = current_splits[0][1]  # Start of first split
            chunk_end = current_splits[-1][2]   # End of last split
            chunks.append((chunk_text, chunk_start, chunk_end))

        return chunks

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        final_chunks = []

        # Get the separator to use
        separator = separators[-1] if separators else ""
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by the separator
        splits = self._split_by_separator(text, separator)

        # Merge splits into chunks
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Split is too large, recursively split with next separator
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    # No more separators, force split by character
                    final_chunks.append(split[:self.chunk_size])
                else:
                    # Try next separator
                    recursive_chunks = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(recursive_chunks)

        # Merge remaining splits
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text by a single separator.

        Args:
            text: Text to split
            separator: Separator to use

        Returns:
            List of split segments
        """
        if separator == "":
            return list(text)

        if self.keep_separator:
            # Keep separator at the end of each split
            splits = []
            pattern = f"({re.escape(separator)})"
            parts = re.split(pattern, text)

            for i in range(0, len(parts) - 1, 2):
                if i + 1 < len(parts):
                    splits.append(parts[i] + parts[i + 1])
                else:
                    splits.append(parts[i])

            # Add the last part if exists
            if len(parts) % 2 == 1:
                splits.append(parts[-1])

            return [s for s in splits if s]
        else:
            return [s for s in text.split(separator) if s]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge small splits into chunks with overlap.

        Args:
            splits: List of text splits to merge
            separator: Separator used for splitting

        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            # Check if adding this split would exceed chunk_size
            if current_length + split_length + len(separator) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = separator.join(current_chunk) if separator != "" else "".join(current_chunk)
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    # Keep last few splits for overlap
                    overlap_length = 0
                    overlap_chunks = []
                    for prev_split in reversed(current_chunk):
                        if overlap_length + len(prev_split) < self.chunk_overlap:
                            overlap_chunks.insert(0, prev_split)
                            overlap_length += len(prev_split) + len(separator)
                        else:
                            break

                    current_chunk = overlap_chunks + [split]
                    current_length = sum(len(s) for s in current_chunk) + len(separator) * (len(current_chunk) - 1)
                else:
                    # Single split is larger than chunk_size
                    current_chunk = [split]
                    current_length = split_length
            else:
                current_chunk.append(split)
                current_length += split_length + (len(separator) if current_chunk else 0)

        # Add the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk) if separator != "" else "".join(current_chunk)
            chunks.append(chunk_text)

        return chunks


class DocumentChunker:
    """
    High-level document chunking interface with metadata preservation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document chunker from configuration.

        Args:
            config: Configuration dictionary with chunking parameters
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('chunk_size', 512),
            chunk_overlap=config.get('chunk_overlap', 50),
            separators=config.get('separators', None),
        )

    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        track_positions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document and preserve metadata.

        Args:
            text: Document text to chunk
            metadata: Document metadata (source, title, date, etc.)
            track_positions: Whether to track character positions in the original
                           document (default: True). Enables snippet-level evaluation.

        Returns:
            List of chunk dictionaries with text, metadata, and character positions
        """
        if metadata is None:
            metadata = {}

        # Use position-aware chunking if requested
        if track_positions:
            chunks_with_pos = self.splitter.split_text_with_positions(text)
        else:
            # Fallback to legacy behavior (backward compatibility)
            chunks_text = self.splitter.split_text(text)
            chunks_with_pos = [(ct, 0, len(ct)) for ct in chunks_text]

        chunk_dicts = []
        for i, (chunk_text, start_char, end_char) in enumerate(chunks_with_pos):
            # For position tracking, use the actual text from the original document
            # at the tracked positions, not the reconstructed chunk_text which may
            # have extra separators from the merging process
            if track_positions and start_char < end_char:
                actual_text = text[start_char:end_char]
            else:
                actual_text = chunk_text

            chunk_dict = {
                'text': actual_text,
                'chunk_id': i,
                'chunk_count': len(chunks_with_pos),
                'start_char': start_char,  # Character position in original document
                'end_char': end_char,      # End position in original document
                **metadata  # Include original metadata
            }
            chunk_dicts.append(chunk_dict)

        return chunk_dicts

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        track_positions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents with 'text' and optional metadata
            track_positions: Whether to track character positions (default: True)

        Returns:
            List of all chunks from all documents

        Note:
            Character positions (start_char, end_char) are relative to each
            individual document, NOT a global corpus position. Each document's
            chunks have positions starting from 0 within that document.
        """
        all_chunks = []

        for doc_id, doc in enumerate(documents):
            text = doc.get('text', '')
            metadata = {k: v for k, v in doc.items() if k != 'text'}
            metadata['doc_id'] = doc_id

            # Positions will be per-document (starting from 0 for each doc)
            chunks = self.chunk_document(text, metadata, track_positions=track_positions)
            all_chunks.extend(chunks)

        return all_chunks


def load_chunker_from_config(config_path: str) -> DocumentChunker:
    """
    Load document chunker from YAML configuration file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configured DocumentChunker instance
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    chunking_config = config.get('chunking', {})
    return DocumentChunker(chunking_config)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    In negligence cases, the plaintiff must establish four elements:

    1. Duty of Care: The defendant owed a duty of care to the plaintiff.

    2. Breach of Duty: The defendant breached that duty through action or inaction.

    3. Causation: The defendant's breach caused the plaintiff's injury.

    4. Damages: The plaintiff suffered actual damages as a result.

    Each element must be proven by a preponderance of the evidence.
    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_text(sample_text)

    print(f"Split into {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} ({len(chunk)} chars):")
        print(chunk)
        print("-" * 80)
