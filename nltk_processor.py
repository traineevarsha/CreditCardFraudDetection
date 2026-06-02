import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download("punkt",        quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("punkt_tab",    quiet=True)

# Fraud-related keywords
FRAUD_KEYWORDS = {
    "suspicious", "unusual", "fraud", "fraudulent", "risk", "high",
    "unauthorized", "anomaly", "irregular", "flagged", "alert",
    "abnormal", "declined", "blocked", "warning"
}

STOP_WORDS = set(stopwords.words("english"))

def tokenize_explanation(text: str) -> dict:
    """
    Task 1 — Tokenization: Split GPT-2 output into words and sentences.
    Returns both word tokens and sentence tokens.
    """
    word_tokens = word_tokenize(text)
    sent_tokens = sent_tokenize(text)
    
    return {
        "words": word_tokens,
        "sentences": sent_tokens
    }

def extract_fraud_keywords(text: str) -> list:
    """
    Task 2 — Keyword Extraction: Filter stopwords and find fraud-indicator words.
    These get highlighted in the Streamlit UI.
    """
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    meaningful_tokens = [t for t in tokens if t not in STOP_WORDS and t.isalpha()]
    
    found_keywords = [t for t in meaningful_tokens if t in FRAUD_KEYWORDS]
    
    return list(set(found_keywords))  # Deduplicate

def clean_gpt2_output(text: str) -> str:
    """
    Task 3 — Cleanup: Remove noise from GPT-2 generated text before display.
    GPT-2 sometimes repeats phrases or produces garbled output.
    """
    sentences = sent_tokenize(text)
    cleaned_sentences = sentences[:3]
    
    cleaned_sentences = [s for s in cleaned_sentences if len(s.split()) >= 5]
    
    return " ".join(cleaned_sentences).strip()

def process_explanation(raw_text: str) -> dict:
    """Run all three NLTK tasks on a GPT-2 explanation."""
    tokens    = tokenize_explanation(raw_text)
    keywords  = extract_fraud_keywords(raw_text)
    clean_txt = clean_gpt2_output(raw_text)
    
    return {
        "tokens":   tokens,
        "keywords": keywords,
        "clean":    clean_txt
    }

if __name__ == "__main__":
    sample = ("This transaction appears suspicious due to an unusual location "
              "and high amount at an irregular hour. The fraud probability is very high. "
              "We recommend blocking this transaction immediately.")
    result = process_explanation(sample)
    print("Clean text:", result["clean"])
    print("Keywords found:", result["keywords"])