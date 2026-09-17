import fitz
import re
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    text: str
    page: int
    is_heading: bool
    font_size: float
    heading_level: int  # 0=body, 1=section, 2=subsection, 3=sub-subsection

def is_likely_heading(text: str, font_size: float, is_bold: bool, base_font_size: float) -> tuple[bool, int]:
    """Determine if a text block is a heading and return its level."""
    text = text.strip()
    if not text or len(text.split()) > 20:
        return False, 0
    
    level = 0
    is_heading = False
    
    # Numbering patterns
    roman_pattern = re.compile(r'^(I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3})\.\s+')
    num_pattern = re.compile(r'^\d+\.\s+')
    letter_pattern = re.compile(r'^[A-Z]\.\s+')
    decimal_pattern = re.compile(r'^\d+\.\d+\.?\s+')

    if roman_pattern.match(text) or num_pattern.match(text) or letter_pattern.match(text):
        is_heading = True
        level = 1
    elif decimal_pattern.match(text):
        is_heading = True
        level = 2
        
    if font_size > base_font_size + 2:
        is_heading = True
        level = 1 if level == 0 else level
    elif font_size > base_font_size + 0.5 or is_bold:
        is_heading = True
        level = 2 if level == 0 else level
        
    if text.isupper():
        is_heading = True
        level = max(1, level)
        
    return is_heading, level

def parse_pdf(pdf_path: str) -> List[TextBlock]:
    logger.info(f"Parsing PDF: {pdf_path}")
    blocks = []
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        raise

    # First pass to find base font size
    font_sizes = []
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["text"].strip():
                            font_sizes.append(span["size"])
                            
    if not font_sizes:
        return []
        
    font_sizes.sort()
    base_font_size = font_sizes[len(font_sizes) // 2]
    
    for page_num, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        for b in page_dict.get("blocks", []):
            if b.get("type") != 0:
                continue
            
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                        
                    font_size = span["size"]
                    font_flags = span["flags"]
                    is_bold = bool(font_flags & 2**4)
                    
                    is_heading, level = is_likely_heading(text, font_size, is_bold, base_font_size)
                    
                    blocks.append(TextBlock(
                        text=text,
                        page=page_num,
                        is_heading=is_heading,
                        font_size=font_size,
                        heading_level=level
                    ))
    doc.close()
    return blocks
