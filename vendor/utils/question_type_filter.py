"""
Utility functions for filtering samples by question type
"""

import re
from typing import List, Dict, Set


def extract_question_type(question: str) -> List[str]:
    """
    Extract question types from question text
    
    Args:
        question: Question text
        
    Returns:
        List of question types (e.g., ['what'], ['where'], ['is', 'spatial'])
    """
    question_lower = question.lower()
    types = []
    
    # Question type patterns
    question_patterns = {
        'what': r'\bwhat\b',
        'where': r'\bwhere\b',
        'how': r'\bhow\b',
        'when': r'\bwhen\b',
        'why': r'\bwhy\b',
        'who': r'\bwho\b',
        'which': r'\bwhich\b',
        'can': r'\bcan\b',
        'is': r'\bis\b',
        'are': r'\bare\b',
        'does': r'\bdoes\b',
        'do': r'\bdo\b',
        'count': r'\bhow many\b',
        'color': r'\bcolor\b',
        'direction': r'\b(left|right|front|back|behind|ahead|direction)\b',
        'yes_no': r'\b(is|are|can|does|do|will|would)\s+\w+',
    }
    
    # Check patterns
    for qtype, pattern in question_patterns.items():
        if re.search(pattern, question_lower, re.IGNORECASE):
            types.append(qtype)
    
    # Special handling
    if 'how many' in question_lower:
        if 'count' not in types:
            types.append('count')
    if any(word in question_lower for word in ['color', 'colour']):
        if 'color' not in types:
            types.append('color')
    if any(word in question_lower for word in ['left', 'right', 'front', 'back', 'behind']):
        if 'spatial' not in types:
            types.append('spatial')
    
    return types if types else ['other']


def filter_samples_by_question_type(
    samples: List[Dict],
    question_types: List[str],
    match_any: bool = True
) -> List[Dict]:
    """
    Filter samples by question type
    
    Args:
        samples: List of sample dictionaries
        question_types: List of question types to filter by (e.g., ['what', 'where'])
        match_any: If True, include samples matching ANY of the types. 
                   If False, include only samples matching ALL types.
        
    Returns:
        Filtered list of samples
    """
    if not question_types:
        return samples
    
    # Normalize question types to lowercase
    question_types = [qtype.lower() for qtype in question_types]
    
    filtered_samples = []
    
    for sample in samples:
        question = sample.get('question', '')
        sample_types = extract_question_type(question)
        sample_types_lower = [t.lower() for t in sample_types]
        
        # Check if sample matches filter criteria
        if match_any:
            # Match if ANY type in sample matches ANY type in filter
            if any(qtype in sample_types_lower for qtype in question_types):
                filtered_samples.append(sample)
        else:
            # Match only if ALL types in filter are in sample
            if all(qtype in sample_types_lower for qtype in question_types):
                filtered_samples.append(sample)
    
    return filtered_samples


def get_question_type_statistics(samples: List[Dict]) -> Dict[str, int]:
    """
    Get statistics on question types in samples
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Dictionary mapping question type to count
    """
    type_counts = {}
    
    for sample in samples:
        question = sample.get('question', '')
        sample_types = extract_question_type(question)
        
        for qtype in sample_types:
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    return type_counts


def split_samples_by_question_type(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Split samples into separate lists by question type
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Dictionary mapping question type to list of samples
    """
    samples_by_type = {}
    
    for sample in samples:
        question = sample.get('question', '')
        sample_types = extract_question_type(question)
        
        # Add sample to each matching type
        for qtype in sample_types:
            if qtype not in samples_by_type:
                samples_by_type[qtype] = []
            samples_by_type[qtype].append(sample)
    
    return samples_by_type

