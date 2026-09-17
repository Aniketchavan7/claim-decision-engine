import tiktoken
import logging
from typing import List
from src.models.evidence import PolicyChunk
from src.ingestion.pdf_parser import TextBlock

logger = logging.getLogger(__name__)

def chunk_document(blocks: List[TextBlock], max_tokens: int = 800, overlap_tokens: int = 100) -> List[PolicyChunk]:
    logger.info("Starting document chunking")
    encoder = tiktoken.get_encoding("cl100k_base")
    
    chunks: List[PolicyChunk] = []
    current_section = ""
    current_subsection = ""
    heading_path: List[str] = []
    
    current_text: List[str] = []
    current_tokens = 0
    current_page = 1
    
    chunk_counter = 0

    def create_chunk(text: str, page: int, sec: str, subsec: str, h_path: List[str]) -> PolicyChunk:
        nonlocal chunk_counter
        chunk_id = f"chunk_{page}_{chunk_counter:03d}"
        chunk_counter += 1
        return PolicyChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            page=page,
            section=sec,
            subsection=subsec,
            heading_path=" > ".join(h_path) if isinstance(h_path, list) else str(h_path)
        )

    for block in blocks:
        if block.is_heading:
            if current_text:
                text_joined = " ".join(current_text)
                chunks.append(create_chunk(text_joined, current_page, current_section, current_subsection, heading_path))
                current_text = []
                current_tokens = 0
                
            if block.heading_level == 1:
                current_section = block.text
                current_subsection = ""
                heading_path = [block.text]
            elif block.heading_level == 2:
                current_subsection = block.text
                if len(heading_path) > 0:
                    heading_path = [heading_path[0], block.text]
                else:
                    heading_path = [block.text]
            else:
                heading_path.append(block.text)
                
            current_text.append(block.text)
            current_tokens += len(encoder.encode(block.text))
            current_page = block.page
            continue
            
        block_tokens = encoder.encode(block.text)
        num_tokens = len(block_tokens)
        
        if current_tokens + num_tokens > max_tokens and current_text:
            text_joined = " ".join(current_text)
            chunks.append(create_chunk(text_joined, current_page, current_section, current_subsection, heading_path))
            
            overlap_text: List[str] = []
            overlap_tok_count = 0
            for prev_text in reversed(current_text):
                prev_toks = len(encoder.encode(prev_text))
                if overlap_tok_count + prev_toks > overlap_tokens:
                    break
                overlap_text.insert(0, prev_text)
                overlap_tok_count += prev_toks
                
            current_text = overlap_text
            current_tokens = overlap_tok_count
            
        current_text.append(block.text)
        current_tokens += num_tokens
        current_page = block.page
        
    if current_text:
        text_joined = " ".join(current_text)
        chunks.append(create_chunk(text_joined, current_page, current_section, current_subsection, heading_path))
        
    return chunks
