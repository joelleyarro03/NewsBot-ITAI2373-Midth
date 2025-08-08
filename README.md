# NewsBot-ITAI2373-Midth
NewsBot project

# 🤖 NewsBot Intelligence System

## 📌 Project Overview

The NewsBot Intelligence System is a group project for the ITAI 2373 course, designed as a complete Natural Language Processing (NLP) pipeline for analyzing news articles.  
Its main goal is to transform raw, unstructured text into structured, actionable insights by performing:

- **Data preprocessing and cleaning**
- **Feature extraction (TF-IDF, POS patterns, syntax features)**
- **Text classification** into predefined categories
- **Sentiment and emotion analysis**
- **Named Entity Recognition (NER)**
- **Comprehensive reporting**

This README provides setup instructions, explains each module step-by-step, and shows how to run the system locally or in Google Colab.

---

## 📂 Table of Contents
1. Introduction]
2. project-overview
3. Setup & Installation]
5. Dataset
6. Project Structure
7. Step-by-Step System Walkthrough
8. Running the Notebook
9. Example Usage
10. Results & Insights
11. Group Contributions
12. References

---

## ⚙ Setup & Installation
You will need **Python 3.8+ and the following dependencies:

```
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn wordcloud plotly networkx
python -m spacy download en_core_web_sm
```

NLTK downloads (run inside Python):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
```

**Colab-specific setup** (if using Google Colab with Kaggle datasets):
```python
from google.colab import files
files.upload()  # upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

## 📊 Dataset
We used the **20 Newsgroups dataset** (via `sklearn.datasets.fetch_20newsgroups`) as a stand-in for real-world news.  
- Articles: ~18,846  
- Categories: 20  
- Features Extracted: TF-IDF vectors (1,000–5,000 terms), POS distributions, sentiment scores, entity counts.

---

## 🗂 Project Structure
```
Final_NewsBot.ipynb      # Main project notebook
README.md                # This documentation
requirements.txt         # List of dependencies
/data                    # Optional storage for local datasets
```

---

## 📜 Step-by-Step System Walkthrough

1. Data Loading & Exploration
- Loaded 20 Newsgroups dataset, merged train/test splits.
- Created DataFrame with `title`, `content`, `category`, `date`, `source`.
- Visualized category distribution.

2. Text Preprocessing
- Lowercasing, removing HTML, URLs, emails, non-alphabetic characters.
- Tokenization (spaCy).
- Stopword removal (NLTK).
- Lemmatization (WordNet).

3. Feature Extraction (TF-IDF)
- Vectorized text with `TfidfVectorizer` (1–2 grams).
- Identified top terms per category.
- Generated word clouds for category visualization.

4. POS Analysis
- Extracted part-of-speech tags per token.
- Compared POS usage across categories.

5. Syntax & Semantic Analysis
- Dependency parsing with spaCy.
- Extracted noun phrases, subjects, and objects.

6. Sentiment Analysis
- Used **NLTK’s VADER** sentiment analyzer.
- Calculated compound, positive, neutral, and negative scores.
- Visualized sentiment distribution by category.

7. Text Classification
- Models: Naive Bayes, Logistic Regression, SVM.
- Best model: **Naive Bayes** (~56.8% accuracy).
- Plotted confusion matrix and performance comparison.

8. Named Entity Recognition (NER)
- Extracted PERSON, ORG, GPE, DATE, etc.
- Counted entities per category.
- Built co-occurrence networks of entities.

9. Final Integration
- Combined preprocessing, classification, sentiment, and NER into one pipeline.
- Generated a final system report.

---

 ▶ Running the Notebook
Local:
```bash
jupyter notebook Final_NewsBot.ipynb
```
Run each cell in sequence.
Google Colab:
- Upload the notebook.
- Install dependencies as shown in setup.
- Run all cells (may take 10–15 minutes).

---

💻 Example Usage
```python
new_title = "Mars Rover Discovers New Rock Formation"
new_content = "NASA announced that the Curiosity rover found a surprising new rock formation on Mars..."
full_text = f"{new_title} {new_content}"

processed = preprocess_text(full_text)
predicted_category = best_model.predict(tfidf_vectorizer.transform([processed]))
entities = extract_entities(full_text)
sentiment = analyze_sentiment(full_text)

print("Predicted category:", predicted_category[0])
print("Entities:", [(e['text'], e['label']) for e in entities])
print("Sentiment:", sentiment['sentiment_label'], sentiment['compound'])
```

---

 📈 Results & Insights
- **Best Classifier:** Naive Bayes (Accuracy ~57%)
- **Most Positive Category:** comp.graphics
- **Most Negative Category:** talk.politics.guns
- **Total Entities Extracted:** 310,810 (102,816 unique)

---

 👥 Group Contributions
- **Data Preprocessing** – Member A
- **Feature Extraction & Visualization** – Member B
- **POS/Syntax Analysis** – Member C
- **Sentiment & Classification** – Member D
- NER & Integration** – Member E

---

📚 References
- scikit-learn Documentation – https://scikit-learn.org/
- spaCy Documentation – https://spacy.io/
- NLTK VADER – https://github.com/cjhutto/vaderSentiment
- 20 Newsgroups Dataset – http://qwone.com/~jason/20Newsgroups/

Contact:
Email: joelleyarro@gmail.com
Github: https://github.com/joelleyarro03/NewsBot-ITAI2373-Midth/edit/main/README.md
