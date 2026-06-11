# Semantic Course Search — Embedding Algorithm Comparison

A Jupyter notebook that builds a **semantic search engine** for course data using Pinecone as a vector database, then benchmarks five embedding approaches ranging from classical NLP to modern transformer models.

## 📌 Project Overview

This project explores how different text embedding algorithms affect the quality of semantic search. Starting from the simplest possible representation (Bag of Words), each algorithm introduces a new level of linguistic understanding — culminating in context-aware transformer embeddings.

All five algorithms are benchmarked against the same course dataset and query set, with results stored in Pinecone using **namespaces** to keep each algorithm isolated within a shared index.

## 🛠️ Libraries Used

- **Vector database:** `pinecone-client`
- **Embeddings:** `sentence-transformers`
- **Classical NLP:** `scikit-learn` (`CountVectorizer`, `TfidfVectorizer`)
- **Data:** `pandas`, `numpy`
- **Environment:** `python-dotenv`

## 🚀 Embedding Algorithms Covered

**1. Bag of Words (BoW)**
- Represents text as raw word occurrence counts
- No semantic understanding — "car" and "automobile" are unrelated
- Baseline for comparison

**2. TF-IDF (Term Frequency–Inverse Document Frequency)**
- Weights words by how unique they are across all documents
- Common words like "the" score low; rare meaningful words score high
- Better signal than BoW, still no true semantic understanding

**3. Word2Vec (averaged token embeddings)**
- Maps individual words to vectors; captures relationships like `king - man + woman ≈ queen`
- Approximated here by averaging token-level embeddings from `all-MiniLM-L6-v2`
- Loses sentence-level context due to averaging

**4. BERT (`multi-qa-distilbert-cos-v1`)**
- Transformer model producing context-aware sentence embeddings
- "bank" in "river bank" vs. "bank account" gets different vectors
- Fine-tuned for semantic search / question-answering tasks

**5. ELMo-style (`all-distilroberta-v1`)**
- ELMo introduced context-aware embeddings via bidirectional LSTMs, predating BERT
- Approximated here with a distilled RoBERTa model — lighter than BERT, still context-aware

## 📊 Key Implementation Details

| Algorithm | Model | Dimension | Pinecone Namespace |
|---|---|---|---|
| Bag of Words | `CountVectorizer` | 768 | `bow` |
| TF-IDF | `TfidfVectorizer` | 768 | `tfidf` |
| Word2Vec | `all-MiniLM-L6-v2` (averaged) | 384 | `w2v` |
| BERT | `multi-qa-distilbert-cos-v1` | 768 | `bert` |
| ELMo-style | `all-distilroberta-v1` | 768 | `elmo` |

**Index strategy:** Rather than creating one Pinecone index per algorithm (which exceeds the free-plan limit of 5 indexes), the notebook uses two shared indexes — `index-768` and `index-384` — with namespaces to logically separate each algorithm's vectors.

## 📂 Repository Structure

```
├── 365 Courses Embedding Comparison.ipynb   # Main notebook — all 5 algorithms
├── course_descriptions.csv                  # Course dataset
├── .env                                     # Environment variable file
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

## 👩‍💻 How to Run

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Pinecone API key:
   ```
   PINECONE_API_KEY=your_key_here
   ```
4. Place `course_descriptions.csv` in the root directory
5. Open `365 Courses Embedding Comparison.ipynb` in Jupyter
6. Run all cells from top to bottom

> **Note:** The first run will download the sentence-transformer models (~250MB total). Subsequent runs reuse the cached models.

## 📈 Sample Queries Used for Comparison

```python
queries = ["clustering", "data science", "python", "finance"]
```

Results are filtered by a cosine similarity threshold of `0.3` and ranked by score. The final cell runs all five algorithms against the same query side by side for direct comparison.

---

**Developed by:**  
Nicole Kaye A. Cardel  
[nkcardel@gmail.com](mailto:nkcardel@gmail.com)  
*Software Designer & Developer*
