# Class 1 & 2: Text Search & Clustering

**Duration**: 2h30 (with 15-minute break at ~1h30)  
**Difficulty**: Introduction to Medium  
**Prerequisites**: Basic Python, pandas, numpy

---

## Quick Navigation

- **[Student Guide](theory/slides/Class%201%20%26%202.md)** - Theory reference with concepts, formulas, and examples
- **[FAQ](theory/slides/FAQ.md)** - Common questions and troubleshooting

---

## Files Structure

```
Class 1 & 2 - NLP and Search/
├── README.md (this file)
├── notebooks/
│   ├── Learning Notebook Part 1.ipynb     ← Foundation: Preprocessing & Bag of Words
│   ├── Learning Notebook Part 2.ipynb     ← Advanced: TF-IDF, Similarity Search & Clustering
│   ├── Learning Notebook Part 3.ipynb     ← Consolidation: Industry Tools (NLTK, spaCy, Hugging Face)
│   ├── Exercise Notebook Part 1.ipynb     ← Class 1 & 2 exercises (follows Learning Part 1)
│   ├── Exercise Notebook Part 2.ipynb     ← Class 1 & 2 exercises (follows Learning Part 2)
│   └── Exercise Notebook Part 3.ipynb     ← Consolidation exercises (follows Learning Part 3)
├── data/
│   └── movies.csv
└── theory/
    └── slides/
        ├── Class 1 & 2.md                  ← Theory reference
        ├── FAQ.md                          ← Troubleshooting
        └── Class 1 & 2.pdf                 ← PDF version
```

---

## Materials

### For Students
1. **Learning Notebook Part 1** - Foundation: Preprocessing & Bag of Words (follow during first part of class)
2. **Exercise Notebook Part 1 (Class 1 & 2)** - Practice exercises for Part 1 (text preprocessing, tokenization, BoW/TF, keyword search)
3. **Learning Notebook Part 2** - Advanced: TF-IDF, Similarity Search & Clustering (follow during second part)
4. **Exercise Notebook Part 2 (Class 1 & 2)** - Practice exercises for Part 2 (TF-IDF, similarity search, RAG)
5. **Learning Notebook Part 3** - Industry Tools: NLTK, spaCy, Hugging Face (consolidation)
6. **Exercise Notebook Part 3** - Practice exercises for Part 3 (NLTK, spaCy, scikit-learn, complete search system)
7. **Student Guide** - Study reference (concepts, formulas, detailed explanations)
8. **FAQ** - Common questions and solutions

**Important**: Exercise notebooks are split into Part 1, Part 2, and Part 3 to align with the learning notebooks. Complete Part 1 exercises before moving to Part 2, and Part 2 before Part 3!

**Note for Exercise Notebook Part 3**: Exercise 12 requires knowledge of Python classes. A quick reference guide is included in the notebook before Exercise 12, with links to learning resources.

### For Instructors
1. **Learning Notebook Part 1 & Part 2** - Main teaching material (split for 45min-1hr sessions)
2. **FAQ** - Common student struggles and solutions
3. **Student Guide (Class 1 & 2.md)** - Complete theory reference for students

---

## Quick Start

### Opening Notebooks in Google Colab

**Option 1: Open from GitHub (Recommended)**
Click the badges below to open notebooks directly in Google Colab:

- [**Learning Notebook Part 1** ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsung-ai-course/8th-9th-edition/blob/main/Chapter%202%20-%20Natural%20Language%20Processing/Class%201%20%26%202%20-%20NLP%20and%20Search/notebooks/Learning%20Notebook%20Part%201.ipynb)
- [**Exercise Notebook Part 1** ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsung-ai-course/8th-9th-edition/blob/main/Chapter%202%20-%20Natural%20Language%20Processing/Class%201%20%26%202%20-%20NLP%20and%20Search/notebooks/Exercise%20Notebook%20Part%201.ipynb)
- [**Learning Notebook Part 2** ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsung-ai-course/8th-9th-edition/blob/main/Chapter%202%20-%20Natural%20Language%20Processing/Class%201%20%26%202%20-%20NLP%20and%20Search/notebooks/Learning%20Notebook%20Part%202.ipynb)
- [**Exercise Notebook Part 2** ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsung-ai-course/8th-9th-edition/blob/main/Chapter%202%20-%20Natural%20Language%20Processing/Class%201%20%26%202%20-%20NLP%20and%20Search/notebooks/Exercise%20Notebook%20Part%202.ipynb)
- [**Learning Notebook Part 3** ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsung-ai-course/8th-9th-edition/blob/main/Chapter%202%20-%20Natural%20Language%20Processing/Class%201%20%26%202%20-%20NLP%20and%20Search/notebooks/Learning%20Notebook%20Part%203.ipynb)
- [**Exercise Notebook Part 3** ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samsung-ai-course/8th-9th-edition/blob/main/Chapter%202%20-%20Natural%20Language%20Processing/Class%201%20%26%202%20-%20NLP%20and%20Search/notebooks/Exercise%20Notebook%20Part%203.ipynb)

**Option 2: Upload to Google Colab (if you have local files)**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload the notebook files from the `notebooks/` folder
4. Upload the `data/movies.csv` file to your Colab session:
   - Click the folder icon 📁 on the left sidebar
   - Click the upload icon (📤) and select `movies.csv` from the `data/` folder
   - Create a `data/` folder in Colab if needed

**⚠️ Important: Data Files**

When using Google Colab (especially from GitHub), you need to upload the data file:

1. **Download the data file** from GitHub:
   - Go to the repository: `data/movies.csv`
   - Click "Raw" and download the file, OR
   - Navigate to: `Chapter 2 - Natural Language Processing/Class 1 & 2 - NLP and Search/data/movies.csv`

2. **Upload to Colab**:
   - In Colab, click the folder icon 📁 on the left sidebar
   - Click the upload icon (📤) and select `movies.csv`
   - Make sure it's in a `data/` folder (create one if needed)

3. **Run the setup cell** in the notebook to install any required packages

### Students

**Part 1 (First ~1 hour):**
1. Open **Learning Notebook Part 1** during first part of class (45min-1hr)
   - Covers: Text preprocessing, tokenization, Bag of Words (BoW/TF)
2. Complete exercises in **Exercise Notebook Part 1 (Class 1 & 2)** after Learning Part 1
   - Exercises: Text cleaning (1a-1d), Tokenization (2a-2e), BoW/TF, keyword search, stemming/lemmatization, advanced regex (9a-9d), special cases (10a-10e)

**Part 2 (Second ~1 hour):**
3. Open **Learning Notebook Part 2** during second part of class (45min-1hr)
   - Covers: TF-IDF, similarity search, document clustering
4. Complete exercises in **Exercise Notebook Part 2 (Class 1 & 2)** after Learning Part 2
   - Exercises: TF-IDF from scratch, similarity-based search, RAG system

**Part 3 (Consolidation - Additional time):**
5. Open **Learning Notebook Part 3** to consolidate learning with industry tools
   - Covers: Industry tools (NLTK, spaCy, Hugging Face, scikit-learn)
6. Complete exercises in **Exercise Notebook Part 3** after Learning Part 3
   - Exercises: NLTK preprocessing, spaCy preprocessing, TF-IDF with custom tokenizers, complete search system with classes
   - **Note**: Exercise 12 requires Python classes knowledge - a quick reference is included in the notebook

**Note**: Document clustering is covered in **Learning Notebook Part 2** but not in exercise notebooks (concepts are sufficient).

**Additional Resources:**
5. Refer to **Student Guide (Class 1 & 2.md)** for theory review
6. Check **FAQ** if stuck

### Instructors
1. Prepare **Learning Notebook Part 1 & Part 2** (test code, load data)
2. Structure class: 45min-1hr theory/showcase → exercises → repeat
3. Keep **FAQ** handy for common questions
4. Use **Student Guide (Class 1 & 2.md)** for complete theory reference

---

## What This Class Covers

**Part 1:**
- Text preprocessing (cleaning with regex, tokenization, edge cases)
- Converting text to numbers (Bag of Words / Term Frequency)
- Keyword search (simple and multiple keywords)
- Advanced preprocessing (stemming, lemmatization, n-grams)
- Advanced regex patterns and special case handling

**Part 2:**
- TF-IDF (Term Frequency-Inverse Document Frequency) calculation
- Similarity-based search using cosine similarity
- Document clustering (K-means on TF-IDF vectors)
- RAG (Retrieval-Augmented Generation) system
- Supervised vs unsupervised learning in NLP

**Part 3 (Consolidation):**
- Industry tools: NLTK, spaCy, Hugging Face
- Using professional NLP libraries for preprocessing
- TF-IDF with custom tokenizers (scikit-learn)
- Building complete search systems with classes
- Documentation and best practices
- Consolidates Parts 1 & 2 by applying concepts with industry-standard tools

**Key Concepts:**
- **BoW/TF**: Simple word counting (foundation for TF-IDF)
- **TF-IDF**: Weighted word importance (syntactic, keyword-based)
- **Similarity Search**: Better than keyword search but still keyword-based
- **Semantic Search**: Requires embeddings (not covered in this class)

*For detailed learning objectives and concept explanations, see the [Student Guide](theory/slides/Class%201%20%26%202.md)*

---

## Exercise Notebooks Overview

### Exercise Notebook Part 1 (Class 1 & 2)
**Follows Learning Notebook Part 1**

**Exercises included:**
- **Exercise 1**: Text Cleaning with Regex (1a-1d)
  - Extract patterns, normalize text, clean HTML/Markdown, compare strategies
- **Exercise 2**: Tokenization (2a-2e)
  - Compare methods, test thresholds, stop words, different text types, n-grams
- **Exercise 3**: Bag of Words (BoW) / Term Frequency (TF) Calculation
- **Exercise 4**: TF-Based Keyword Search
- **Exercise 7**: Compare Preprocessing Approaches
- **Exercise 8**: Stemming and Lemmatization (8a-8d)
- **Exercise 9**: Advanced Regex Patterns (9a-9d)
  - Dates, currency, lookahead/lookbehind, substitution with callbacks
- **Exercise 10**: Handling Special Cases (10a-10e)
  - Unicode, capitalization, missing data, numbers/units, robust pipeline

### Exercise Notebook Part 2 (Class 1 & 2)
**Follows Learning Notebook Part 2**

**Exercises included:**
- **Exercise 3**: TF-IDF from Scratch (IDF calculation + full TF-IDF)
- **Exercise 5**: Similarity-Based Search with TF-IDF
- **Exercise 11**: RAG (Retrieval-Augmented Generation) with Movies

**Note**: Document clustering exercises are covered in **Learning Notebook Part 2** and not included in exercise notebooks.

### Exercise Notebook Part 3
**Follows Learning Notebook Part 3** (Consolidation)

**Exercises included:**
- **Exercise 8**: Text Preprocessing with NLTK
- **Exercise 9**: Text Preprocessing with spaCy
- **Exercise 10**: TF-IDF with Custom Tokenizers
- **Exercise 12**: Complete Search System with Industry Tools
  - **Note**: Requires Python classes knowledge - quick reference guide included in notebook
- **Exercise 13**: Compare Tools Side-by-Side

---

## Resources

- [Student Guide](theory/slides/Class%201%20%26%202.md) - Detailed theory reference
- [FAQ](theory/slides/FAQ.md) - Questions & answers
