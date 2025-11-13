# Class 3: Word Embeddings & Sentiment Analysis

**Difficulty**: Medium  
**Prerequisites**: Class 1 & 2 (TF-IDF, vectors, preprocessing, Logistic Regression from Chapter 0)

---

## Quick Navigation

- **[Theory Guide](theory/slides/Class%203.md)** - Complete theory reference with concepts and examples
- **[FAQ](theory/slides/FAQ.md)** - Common questions and troubleshooting

---

## Class Structure

1. **Slides Block 1** - Theory foundations (Sparse to Dense, Word2Vec, GloVe, FastText)
2. **Learning Notebook Part 1** - Word embeddings, visualization, semantic search
3. **Break**
4. **Slides Block 2** - Advanced concepts (Training Objectives, Linguistic Features)
5. **Learning Notebook Part 2** - Sentence embeddings & classification
6. **Wrap-up Discussion**
7. **Buffer/Q&A**

---

## Files Structure

```
Class 3 - Understanding Embeddings/
├── README.md (this file)
├── notebooks/
│   ├── Learning Notebook Part 1.ipynb  ← Word embeddings, visualization, semantic search (Word2Vec, GloVe, FastText)
│   └── Learning Notebook Part 2.ipynb  ← Sentence embeddings, training objectives, classification (BERT, Sentence-BERT)
├── data/
│   └── IMDB Dataset.csv (50,000 movie reviews)
└── theory/
    └── slides/
        ├── Class 3.md           ← Complete theory reference
        └── FAQ.md               ← Common questions and troubleshooting
```

---

## Materials

### For Students
1. **Learning Notebook Part 1** - Word embeddings introduction, visualization, semantic search (Word2Vec, GloVe, FastText)
2. **Learning Notebook Part 2** - Sentence embeddings, training objectives, classification (BERT, Sentence-BERT)
3. **Theory Guide (Class 3.md)** - Complete reference with concepts, explanations, and visualizations
4. **FAQ** - Common questions and troubleshooting

### For Instructors
1. **Learning Notebooks** - Main teaching material (test before class!)
2. **Theory Guide (Class 3.md)** - Complete content reference
3. **FAQ** - Common student struggles and solutions
4. **Slides** - Prepare slides based on theory guide (emphasize visualization and intuition!)

---

## What This Class Covers

### Part 1: Word Embeddings & Visualization
- **Recap**: From sparse (TF-IDF) to dense (embeddings) - solving the synonym problem
- **Word2Vec**: Local context-based embeddings - explore similarities, vector arithmetic, OOV handling
- **GloVe**: Global co-occurrence statistics - explore similarities, compare with Word2Vec
- **FastText**: Subword information - better OOV handling (its key advantage!)
- **Visualization**: t-SNE plots for Word2Vec, GloVe, FastText - see how similar words cluster!
- **Preprocessing Impact**: How preprocessing affects word similarities and semantic search
- **Semantic Search**: Test all embedding methods with real search queries - see semantic understanding in action!

**Key Insight**: Part 1 introduces word embeddings and visualizes them - no supervised learning yet!

### Part 2: Sentence Embeddings & Classification
- **Baseline**: Quick word averaging for classification (from Part 1)
- **How Sentence/Document Embeddings Are Learned**: Three training objectives (conceptually explained!)
  - **Next-token prediction** (GPT style) - predicting what's next → adjusts vectors to understand context
  - **Masked Language Modeling** (BERT style) - predicting what's missing → adjusts vectors to understand bidirectional context
  - **Contrastive learning** (Sentence-BERT style) - predicting what's similar → adjusts vectors to push similar together, dissimilar apart
- **Key Concept**: **LEARN = Adjust the Vectors** - learning means adjusting vector representations based on training objectives
- **Sentence-BERT**: Modern sentence embeddings - better than word averaging!
- **Classification**: Use Sentence-BERT for sentiment analysis - compare with word averaging baseline

**Key Insight**: Part 2 focuses on sentence-level embeddings and how they're learned. All three training methods require understanding context first, then adjust vectors to improve predictions. Sentence-BERT (trained with contrastive learning) outperforms word averaging for classification!

---

## Technical Requirements

### Software/Libraries

```python
pandas
numpy
scikit-learn
gensim
spacy
sentence-transformers
matplotlib
seaborn
nltk
```

### Pre-download Commands

**spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

**NLTK data:**
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

**Gensim models** (downloaded automatically in notebooks):
- Word2Vec: `word2vec-google-news-300` (~1.6GB)
- GloVe: `glove-wiki-gigaword-300` (~400MB)
- FastText: `fasttext-wiki-news-subwords-300` (~900MB)

**Sentence-BERT** (downloaded automatically):
- Model: `all-MiniLM-L6-v2` (~90MB)

### Dataset

**IMDB Movie Reviews**
- Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- File: `IMDB Dataset.csv`
- Size: 50,000 reviews
- Format: CSV with 'review' and 'sentiment' columns

---

## Pre-Class Checklist

### 1 Week Before
- [ ] Download **IMDB dataset** from Kaggle
- [ ] Review **Slides Block 1** content (15-20 slides)
- [ ] Review **Slides Block 2** content (10-15 slides)
- [ ] Review **Learning Notebook Part 1** (embeddings comparison)
- [ ] Review **Learning Notebook Part 2** (advanced techniques)

### 3 Days Before
- [ ] Test **Learning Notebook Part 1** end-to-end
- [ ] Test **Learning Notebook Part 2** end-to-end
- [ ] Verify all models download correctly:
  - [ ] Word2Vec (gensim) - first run will download ~1.6GB
  - [ ] GloVe (gensim) - first run will download ~400MB
  - [ ] FastText (gensim) - first run will download ~900MB
  - [ ] Sentence-BERT - first run will download ~90MB
  - [ ] spaCy model (`en_core_web_sm`)
- [ ] Check all visualizations render properly
- [ ] Prepare 5 wrap-up discussion questions

### 1 Day Before
- [ ] Upload notebooks to platform (Colab/JupyterHub/etc)
- [ ] Test notebooks on platform (verify models download)
- [ ] Export slides to PDF (if using slides)
- [ ] Prepare demo links (HuggingFace spaces for training objectives)
- [ ] Send access links to students

### Day Of
- [ ] Test internet connection (models download on first run)
- [ ] Open all demo links in browser tabs (HuggingFace)
- [ ] Have slides ready
- [ ] Have notebooks open and tested

---

## Learning Objectives

By the end of this class, students will:

1. **Understand the semantic leap**: How embeddings solve the synonym problem from Class 2
2. **Compare word embedding methods**: Word2Vec, GloVe, FastText - strengths and weaknesses
3. **Visualize embeddings**: Use t-SNE to see how similar words cluster in embedding space
4. **Understand how embeddings are learned**: Three training objectives (conceptually explained!)
   - Next-token prediction (GPT style) - predicting what's next
   - Masked Language Modeling (BERT style) - predicting what's missing
   - Contrastive learning (Sentence-BERT style) - predicting what's similar
5. **Key concept**: **LEARN = Adjust the Vectors** - learning means adjusting vector representations based on training objectives
6. **Understand sentence embeddings**: BERT and Sentence-BERT - better than word averaging!
7. **Apply sentence embeddings**: Use Sentence-BERT for sentiment classification

**Key Concept**: **Semantic = meaning**. Embeddings understand meaning, not just word patterns! And sentence embeddings learn by adjusting vectors based on training objectives - all three methods require understanding context first!

---

## Quick Start

### Students
1. Open **Learning Notebook Part 1** during first half of class
2. Follow along with instructor through word embeddings, visualization, and semantic search
3. Open **Learning Notebook Part 2** during second half
4. Learn about sentence embeddings, training objectives, and classification
5. Refer to **Theory Guide (Class 3.md)** for detailed explanations
6. Check **FAQ** if stuck

### Instructors
1. Review **Theory Guide (Class 3.md)** for complete content
2. Prepare **Learning Notebooks** (test code, verify models download)
3. Keep **FAQ** handy for common questions
4. Emphasize visualization and intuition - t-SNE plots, vector arithmetic, analogies
5. Allocate extra time for embedding concept explanation (Word2Vec, GloVe, FastText need reflection time!)
6. Explain training objectives conceptually - students understand HOW embeddings are learned, not training them

---

## Resources

- **[Theory Guide](theory/slides/Class%203.md)** - Complete theory reference with concepts, visualizations, and explanations
- **[FAQ](theory/slides/FAQ.md)** - Common questions and troubleshooting
- **[Class 1 & 2](../Class%201%20%26%202%20-%20NLP%20and%20Search/README.md)** - Prerequisites (TF-IDF, preprocessing, semantic search)
- **Chapter 0** - Logistic Regression (used for classification in Part 2)

---

## Notes for Instructors

### Key Teaching Points

1. **Emphasize the semantic leap**: This class solves the limitation students learned about in Class 2
2. **Visualization is crucial**: Use t-SNE plots, vector arithmetic examples, analogies (especially in Part 1)
3. **Word embeddings need reflection time**: Visualization and intuition are key for Part 1
4. **Sentence embeddings need clear explanation**: Training objectives need conceptual explanation in Part 2
5. **Training objectives**: Explain conceptually - students understand HOW embeddings are learned (adjusting vectors), not training them
6. **Key concept**: **LEARN = Adjust the Vectors** - emphasize this throughout Part 2
7. **Supervised learning context**: Briefly explain metrics in Part 2 (they'll be covered in depth later)

### Common Student Questions

- "Why do embeddings work better than TF-IDF?" → They capture semantic meaning
- "Which embedding method should I use?" → Depends on use case (see comparison section)
- "How are embeddings learned?" → Three methods: next-token prediction, masked language modeling, contrastive learning - all adjust vectors based on objectives
- "What's the difference between word and sentence embeddings?" → Sentence embeddings capture word order and context, learned through training objectives
- "What does 'LEARN = Adjust the Vectors' mean?" → Learning means changing the numbers in embeddings based on what we want to predict

---

## References

- **Word2Vec**: Mikolov et al., 2013 - "Efficient Estimation of Word Representations in Vector Space"
- **GloVe**: Pennington et al., 2014 - "Global Vectors for Word Representation"
- **FastText**: Bojanowski et al., 2017 - "Enriching Word Vectors with Subword Information"
- **BERT**: Devlin et al., 2018 - "BERT: Pre-training of Deep Bidirectional Transformers"
- **GPT**: Radford et al., 2019 - "Language Models are Unsupervised Multitask Learners"
- **Sentence-BERT**: Reimers & Gurevych, 2019 - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

---

## Next Steps

After Class 3, students understand:
- ✅ Syntactic vs semantic representations
- ✅ How embeddings solve the synonym problem
- ✅ Different word embedding methods (Word2Vec, GloVe, FastText) and when to use them
- ✅ How sentence/document embeddings are learned (three training objectives conceptually)
- ✅ Key concept: **LEARN = Adjust the Vectors** - learning means adjusting representations based on objectives
- ✅ Sentence embeddings (BERT, Sentence-BERT) vs word averaging
- ✅ How to use sentence embeddings for text classification

**Coming next**: Neural networks and deep learning (where embeddings come from - the architectures behind these training objectives!)
