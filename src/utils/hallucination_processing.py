import re
import jiwer
from collections import Counter

def detect_hallucinations(transcript):
    words = transcript.split()
    word_counts = Counter(words)
    word_length = len(words)
    
    for word,count in word_counts.items():
        if 3 <= word_length <= 20:
            if (count/word_length) > 0.3:
                return True
        elif 20 < word_length <= 80:
            if (count/word_length) > 0.2:
                return True
        elif word_length > 80:
            if (count/word_length) > 0.1:
                return True
        
    return False

def generate_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def remove_repetitive_ngrams(transcript, n):
    words = transcript.split()
    ngram_list = [' '.join(words[i:i+n]) for i in range(0, len(words), n)]
#     ngram_list = generate_ngrams(transcript, n)
    cleaned_ngrams = []
    
    if len(ngram_list) > 0:
        cleaned_ngrams = [ngram_list[0]]
    for ngram in ngram_list:
        if (cleaned_ngrams[-1] != ngram):
            cleaned_ngrams.append(ngram)
    
    
    cleaned_transcript = ' '.join(cleaned_ngrams)
    
    return cleaned_transcript.strip()

def remove_hallucinations(transcript):
    for n in range(30, 0, -1):
        transcript = remove_repetitive_ngrams(transcript, n)
    #to handle things like 'o, o o,' where o, is technically not the same as o.
    pattern = re.compile(r'\b(?:[oe],? ?)+\b', re.IGNORECASE)
    cleaned_transcript = pattern.sub('', transcript)
    cleaned_transcript = re.sub(' +', ' ', cleaned_transcript)
    return cleaned_transcript.strip()
    return transcript
   
