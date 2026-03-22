import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def setup_nltk():
    for pkg in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg)

setup_nltk()

class TextCleaner:
    def __init__(
        self,
        language='english',
        remove_stopwords=True,        
        remove_punctuation=True,     
        remove_numbers=True          
    ):
        self.stop_words = set(stopwords.words(language))
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))  
        
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)  
        
        tokens = word_tokenize(text)
        
        if self.remove_stopwords:
            tokens = [
                word for word in tokens 
                if word not in self.stop_words and word.strip() != ''
            ]  
        else:
            tokens = [word for word in tokens if word.strip() != '']

        return " ".join(tokens)
        
    def clean_corpus(self, corpus):
        return [self.clean_text(doc) for doc in corpus]
