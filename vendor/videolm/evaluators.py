"""
NLP-based evaluation methods for VideoQA
"""

import re
from typing import Dict, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback to old punkt if punkt_tab not available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from Levenshtein import ratio as levenshtein_ratio
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False


class AnswerEvaluator:
    """Evaluate answers using multiple NLP methods"""
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        semantic_threshold: float = 0.7,
        bleu_threshold: float = 0.3,
        rouge_threshold: float = 0.3,
        fuzzy_threshold: float = 0.7
    ):
        """
        Initialize evaluator
        
        Args:
            embedding_model: Sentence transformer model name
            semantic_threshold: Threshold for semantic similarity (0-1)
            bleu_threshold: Threshold for BLEU score (0-1)
            rouge_threshold: Threshold for ROUGE score (0-1)
            fuzzy_threshold: Threshold for fuzzy matching (0-1)
        """
        self.semantic_threshold = semantic_threshold
        self.bleu_threshold = bleu_threshold
        self.rouge_threshold = rouge_threshold
        self.fuzzy_threshold = fuzzy_threshold
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Warning: Could not load embedding model {embedding_model}: {e}")
        
        # Initialize ROUGE scorer
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            except Exception as e:
                print(f"Warning: Could not initialize ROUGE scorer: {e}")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower().strip()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def semantic_similarity(self, pred: str, gt: str) -> Tuple[float, bool]:
        """
        Compute semantic similarity using embeddings
        
        Returns:
            (similarity_score, is_correct)
        """
        if not self.embedding_model:
            return 0.0, False
        
        try:
            # Get embeddings
            pred_emb = self.embedding_model.encode([self.normalize_text(pred)])[0]
            gt_emb = self.embedding_model.encode([self.normalize_text(gt)])[0]
            
            # Compute cosine similarity
            similarity = np.dot(pred_emb, gt_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(gt_emb))
            
            is_correct = similarity >= self.semantic_threshold
            return float(similarity), is_correct
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0, False
    
    def bleu_score(self, pred: str, gt: str) -> Tuple[float, bool]:
        """
        Compute BLEU score
        
        Returns:
            (bleu_score, is_correct)
        """
        if not BLEU_AVAILABLE:
            return 0.0, False
        
        try:
            # Tokenize
            pred_tokens = word_tokenize(self.normalize_text(pred))
            gt_tokens = word_tokenize(self.normalize_text(gt))
            
            # Compute BLEU with smoothing
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothing)
            
            is_correct = score >= self.bleu_threshold
            return float(score), is_correct
        except Exception as e:
            print(f"Error in BLEU calculation: {e}")
            return 0.0, False
    
    def rouge_score(self, pred: str, gt: str) -> Tuple[Dict[str, float], bool]:
        """
        Compute ROUGE scores
        
        Returns:
            (rouge_scores_dict, is_correct)
        """
        if not self.rouge_scorer:
            return {}, False
        
        try:
            scores = self.rouge_scorer.score(self.normalize_text(gt), self.normalize_text(pred))
            
            # Use ROUGE-L F1 score as main metric
            rouge_l_f1 = scores['rougeL'].fmeasure
            rouge1_f1 = scores['rouge1'].fmeasure
            
            is_correct = rouge_l_f1 >= self.rouge_threshold
            
            return {
                'rouge1': rouge1_f1,
                'rougeL': rouge_l_f1
            }, is_correct
        except Exception as e:
            print(f"Error in ROUGE calculation: {e}")
            return {}, False
    
    def fuzzy_match(self, pred: str, gt: str) -> Tuple[float, bool]:
        """
        Compute fuzzy string matching using Levenshtein distance
        
        Returns:
            (similarity_ratio, is_correct)
        """
        if not LEVENSHTEIN_AVAILABLE:
            return 0.0, False
        
        try:
            pred_norm = self.normalize_text(pred)
            gt_norm = self.normalize_text(gt)
            
            ratio = levenshtein_ratio(pred_norm, gt_norm)
            is_correct = ratio >= self.fuzzy_threshold
            
            return float(ratio), is_correct
        except Exception as e:
            print(f"Error in fuzzy matching: {e}")
            return 0.0, False
    
    def contains_match(self, pred: str, gt: str) -> Tuple[float, bool]:
        """
        Check if ground truth answer is contained in predicted answer
        Since GT answers are typically short, this checks if GT words/phrase appear in prediction
        
        Returns:
            (match_score, is_correct)
        """
        try:
            pred_norm = self.normalize_text(pred)
            gt_norm = self.normalize_text(gt)
            
            # Check if entire GT string is in prediction
            if gt_norm in pred_norm:
                return 1.0, True
            
            # Check if all words from GT are in prediction
            gt_words = gt_norm.split()
            if len(gt_words) > 0:
                pred_words = set(pred_norm.split())
                matched_words = sum(1 for word in gt_words if word in pred_words)
                match_score = matched_words / len(gt_words)
                
                # Consider correct if all words match or if single word and it matches
                is_correct = match_score >= 0.8 or (len(gt_words) == 1 and match_score > 0)
                
                return float(match_score), is_correct
            
            return 0.0, False
        except Exception as e:
            print(f"Error in contains matching: {e}")
            return 0.0, False
    
    def evaluate(self, pred: str, gt: str) -> Dict:
        """
        Evaluate prediction against ground truth using all methods
        
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Semantic similarity
        sem_score, sem_correct = self.semantic_similarity(pred, gt)
        results['correct_semantic'] = bool(sem_correct)  # Ensure Python bool
        results['semantic_similarity'] = float(sem_score)  # Ensure Python float
        
        # BLEU score
        bleu_score, bleu_correct = self.bleu_score(pred, gt)
        results['correct_bleu'] = bool(bleu_correct)  # Ensure Python bool
        results['bleu_score'] = float(bleu_score)  # Ensure Python float
        
        # ROUGE scores
        rouge_scores, rouge_correct = self.rouge_score(pred, gt)
        results['correct_rouge'] = bool(rouge_correct)  # Ensure Python bool
        if rouge_scores:
            results['rouge1'] = float(rouge_scores.get('rouge1', 0.0))  # Ensure Python float
            results['rougeL'] = float(rouge_scores.get('rougeL', 0.0))  # Ensure Python float
        
        # Fuzzy matching
        fuzzy_score, fuzzy_correct = self.fuzzy_match(pred, gt)
        results['correct_fuzzy'] = bool(fuzzy_correct)  # Ensure Python bool
        results['fuzzy_similarity'] = float(fuzzy_score)  # Ensure Python float
        
        # Contains matching (GT answer included in prediction)
        contains_score, contains_correct = self.contains_match(pred, gt)
        results['correct_contains'] = bool(contains_correct)  # Ensure Python bool
        results['contains_score'] = float(contains_score)  # Ensure Python float
        
        return results

