# MachineLearning_TextModuleTHVK
Text classification pipeline: EDA → TF-IDF &amp; BERT/SBERT embeddings → training &amp; evaluatio

# Folder structure
    text-ml-project/
    │
    ├── README.md
    ├── requirements.txt
    ├── config.py
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── features/
    │   ├── tfidf/
    │   └── bert/
    │
    ├── models/
    │
    ├── results/
    │   ├── figures/
    │   ├── tables/
    │   └── logs/
    │
    ├── modules/
    │   ├── __init__.py
    │   │
    │   ├── data_loader.py
    │   ├── preprocess.py
    │   ├── bert_embed.py
    │   ├── tfidf_features.py
    │   ├── train_classical.py
    │   ├── metrics.py
    │ 
    │
    ├── notebooks/
    │   ├── 01_eda.ip
    │   └── final.ipynb
    │
    └── report/
        └── report.tex


# Pipeline
1. Data understanding
    - Load dataset
    - EDA
        + Label distribution
        + Text length
        + Word frequency
2. Data preparation 
    - Train/test split
    - Preprocessing
        + Lowercase
        + Remove punctuation
        + remove stopwords
        + tokenization

3. Feature Extraction
    - TF-IDF / n-gram
    - BERT embeddings
    - Save embeddings (.npy format)

4. Modeling
    - Train classifier 
        + Naive Bayes
        + Logistic Regression
        + SVM
    - Hyperparameter tuning

5. Evaluation & analysis
   - accuracy
   - precision
   - recall
   - F1
   - confusion matrix