import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tải các gói dữ liệu cần thiết từ nltk (Chỉ tải nếu máy chưa có)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextCleaner:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # 1. Chuyển thành chữ thường
        text = text.lower()
        
        # 2. Loại bỏ dấu câu và số (tùy chọn theo ngữ cảnh dữ liệu)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4. Loại bỏ stopwords và các khoảng trắng thừa
        cleaned_tokens = [word for word in tokens if word not in self.stop_words and word.strip() != '']
        
        # Nối lại thành chuỗi đã làm sạch
        return " ".join(cleaned_tokens)
        
    def clean_corpus(self, corpus):
        """Áp dụng làm sạch cho một list các văn bản."""
        return [self.clean_text(doc) for doc in corpus]
