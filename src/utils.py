"""
Utility functions module
"""
import base64
import os
import re
from pathlib import Path
from datetime import datetime


def encode_image_to_base64(image_path):
    """
    Encode an image to a base64 string.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Base64 encoded string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ensure_dir(directory):
    """
    Ensure a directory exists; create it if it does not.
    
    Args:
        directory: Directory path.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_label_from_filename(filename):
    """
    Extract label from filename.
    Rule: Files ending with _0.jpg are clean(0), files ending with _1.jpg are dirty(1).
    
    Args:
        filename: Filename string.
        
    Returns:
        0 (clean) or 1 (dirty).
    """
    if filename.endswith('_0.jpg'):
        return 0
    elif filename.endswith('_1.jpg'):
        return 1
    else:
        raise ValueError(f"Unable to extract label from filename: {filename}")


def parse_model_response(response_text):
    """
    Robustly parse model response to extract clean or dirty labels.
    
    Handles:
    1. Complete/incomplete <think> tags.
    2. Long text with reasoning steps.
    3. Leading/trailing whitespace.
    4. Case variations.
    
    Args:
        response_text: Text returned by the model.
        
    Returns:
        0 (clean), 1 (dirty), or -1 (unparsable).
    """
    if not response_text or not isinstance(response_text, str):
        return -1
    
    # ========== Step 1: Clean think tags (complete and incomplete) ==========
    # Remove complete <think>...</think> tags
    text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove incomplete closing tags
    text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)
    
    # Remove incomplete opening tags (where no closing tag exists)
    if re.search(r'<think>', text, flags=re.IGNORECASE):
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # ========== Step 2: Basic cleaning ==========
    text = text.strip()
    
    # ========== Step 3: Extract the last valid word ==========
    # Strategy: Many models provide the answer at the end, even after long reasoning.
    lines = text.split('\n')
    last_line = lines[-1].strip() if lines else text
    
    # Try to extract the very last word from the last line
    last_words = last_line.split()
    last_word = last_words[-1].strip('.,!?;:*"\'') if last_words else ''
    
    # ========== Step 4: Multi-strategy matching ==========
    
    # Keyword sets (English variants)
    CLEAN_WORDS = ['clean', '0', 'clear', 'spotless', 'undusty']
    DIRTY_WORDS = ['dirty', '1', 'soiling', 'dusty', 'soiled', 'dust', 'polluted']

    # Strategy 1: Check the last word (Highest priority)
    last_word_lower = last_word.lower()
    if any(w in last_word_lower for w in CLEAN_WORDS):
        return 0
    elif any(w in last_word_lower for w in DIRTY_WORDS):
        return 1
    
    # Strategy 2: Check the last line
    last_line_lower = last_line.lower()
    if any(w in last_line_lower for w in CLEAN_WORDS):
        return 0
    elif any(w in last_line_lower for w in DIRTY_WORDS):
        return 1
    
    # Strategy 3: Check the entire text (lowercase) with boundary matching
    text_lower = text.lower()
    
    has_clean = any(re.search(rf'\b{w}\b', text_lower) for w in CLEAN_WORDS)
    has_dirty = any(re.search(rf'\b{w}\b', text_lower) for w in DIRTY_WORDS)

    if has_clean and not has_dirty:
        return 0
    elif has_dirty and not has_clean:
        return 1
    
    # Strategy 4: Look for answers surrounded by special markers (e.g., **dirty**)
    marked_match = re.search(r'\*\*(clean|dirty)\*\*', text_lower)
    if marked_match:
        return 0 if marked_match.group(1) == 'clean' else 1
    
    # Strategy 5: Look for content following "Answer:" or "Conclusion:"
    answer_match = re.search(r'(?:answer|conclusion|result|label)[:]\s*(clean|dirty)', text_lower)
    if answer_match:
        return 0 if answer_match.group(1) == 'clean' else 1
    
    # Strategy 6: Handle contradictory outputs (if both words appear)
    # Strategy: Determine which keyword appears closest to the end of the text.
    last_clean_pos = -1
    for w in CLEAN_WORDS:
        pos = text_lower.rfind(w)
        if pos > last_clean_pos: 
            last_clean_pos = pos
        
    last_dirty_pos = -1
    for w in DIRTY_WORDS:
        pos = text_lower.rfind(w)
        if pos > last_dirty_pos: 
            last_dirty_pos = pos
    
    if last_clean_pos != -1 and last_dirty_pos != -1:
        # Contradiction handling: Pick the one that appeared last
        return 0 if last_clean_pos > last_dirty_pos else 1
    elif last_clean_pos != -1:
        return 0
    elif last_dirty_pos != -1:
        return 1
    
    # ========== Step 5: Digit matching (Fallback) ==========
    # Look for isolated '0' or '1'
    if re.search(r'\b0\b', text):
        return 0
    elif re.search(r'\b1\b', text):
        return 1
    
    # ========== Unparsable ==========
    return -1


def format_timestamp(dt=None):
    """
    Format a timestamp for filenames or logging.
    
    Args:
        dt: datetime object, defaults to current time if None.
        
    Returns:
        Formatted time string (YYYYMMDD_HHMMSS).
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")