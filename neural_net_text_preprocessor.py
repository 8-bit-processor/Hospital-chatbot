import re
from typing import List, Tuple, Optional
import nltk.downloader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import PorterStemmer
import nltk
import os

# Create a directory for NLTK data
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add the local NLTK data directory to the search path
nltk.data.path.append(nltk_data_dir)

# Download the 'punkt' tokenizer if not already present
try:
    nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
except nltk.downloader.download_error:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', download_dir=nltk_data_dir)

class TextPreprocessor:
    """
    Preprocesses a text corpus for neural network training, including:
    - Sentence splitting
    - Normalization (lowercasing, punctuation removal, etc.)
    - Filtering short/non-alphabetic sentences
    - Vectorization (TF-IDF)
    - Optional stemming
    """
    def __init__(self, min_sentence_length: int = 3, remove_punctuation: bool = True, use_tfidf: bool = True, use_bigram: bool = False, use_stemming: bool = False, min_df: int = 1, max_df: float = 1.0):
        self.min_sentence_length = min_sentence_length
        self.remove_punctuation = remove_punctuation
        self.use_tfidf = use_tfidf
        self.use_bigram = use_bigram
        self.use_stemming = use_stemming
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer: Optional[TfidfVectorizer] = None
        if self.use_stemming:
            self.stemmer = PorterStemmer()

    def split_sentences(self, text: str) -> List[str]:
        # Use NLTK for more robust sentence splitting
        return nltk.sent_tokenize(text)

    def normalize_sentence(self, sentence: str) -> str:
        s = sentence.lower()
        if self.remove_punctuation:
            s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        if self.use_stemming:
            s = ' '.join([self.stemmer.stem(word) for word in s.split()])
        return s

    def filter_sentences(self, sentences: List[str]) -> List[str]:
        filtered = []
        for s in sentences:
            norm = self.normalize_sentence(s)
            if len(norm.split()) >= self.min_sentence_length and any(c.isalpha() for c in norm):
                filtered.append(norm)
        return filtered

    def prepare_sequence_pairs(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        X, y = [], []
        for i in range(len(sentences) - 1):
            if sentences[i] and sentences[i+1]:
                X.append(sentences[i])
                y.append(sentences[i+1])
        return X, y

    def vectorize(self, X_sentences: List[str], y_sentences: List[str]) -> Tuple:
        # Combine all sentences to ensure consistent vocabulary
        all_sentences = X_sentences + y_sentences
        
        if self.use_tfidf:
            vectorizer_class = TfidfVectorizer
        else:
            vectorizer_class = CountVectorizer

        if self.use_bigram:
            self.vectorizer = vectorizer_class(ngram_range=(1, 2), min_df=self.min_df, max_df=self.max_df)
        else:
            self.vectorizer = vectorizer_class(min_df=self.min_df, max_df=self.max_df)
        
        self.vectorizer.fit(all_sentences)
        
        # Transform using the same fitted vectorizer
        X = self.vectorizer.transform(X_sentences).toarray()
        y = self.vectorizer.transform(y_sentences).toarray()
        
        # Ensure same number of samples
        min_samples = min(X.shape[0], y.shape[0])
        return X[:min_samples, :], y[:min_samples, :]

    def preprocess_text_for_sequence(self, text: str) -> Tuple:
        sentences = self.split_sentences(text)
        filtered = self.filter_sentences(sentences)
        if len(filtered) < 2:
            raise ValueError("Not enough valid sentences for sequence training.")
        X_sent, y_sent = self.prepare_sequence_pairs(filtered)
        return self.vectorize(X_sent, y_sent)

    def preprocess_text_for_qa(self, qa_pairs: List[Tuple[str, str]]) -> Tuple:
        # For Q/A, vectorize questions and answers using consistent encoding
        questions = [self.normalize_sentence(q) for q, a in qa_pairs]
        answers = [self.normalize_sentence(a) for q, a in qa_pairs]
        
        # Combine questions and answers to ensure consistent vocabulary
        all_text = questions + answers
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(all_text)
        
        # Transform using the same fitted vectorizer
        X = self.vectorizer.transform(questions).toarray()
        y = self.vectorizer.transform(answers).toarray()
        
        # Ensure same dimensions
        min_samples = min(X.shape[0], y.shape[0])
        return X[:min_samples, :], y[:min_samples, :]

    def encode_text(self, text: str):
        """
        Encode new text using the same vectorizer that was used during training.
        This ensures consistent encoding format for prediction.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Please train the model first.")
        normalized = self.normalize_sentence(text)
        return self.vectorizer.transform([normalized]).toarray()
    
    def decode_vector(self, vector, max_words=10):
        """
        Decode a vector back to text representation (best effort).
        Note: This is approximate since TF-IDF transformation is not perfectly reversible.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Please train the model first.")
        
        # Get feature names and find top features
        feature_names = self.vectorizer.get_feature_names_out()
        if len(vector.shape) > 1:
            vector = vector.flatten()
        
        # Get indices of non-zero elements sorted by value
        non_zero_indices = vector.nonzero()[0]
        if len(non_zero_indices) == 0:
            return ""
        
        # Sort by TF-IDF score and take top words
        sorted_indices = non_zero_indices[vector[non_zero_indices].argsort()[::-1]]
        top_words = [feature_names[i] for i in sorted_indices[:max_words]]
        
        return " ".join(top_words)

