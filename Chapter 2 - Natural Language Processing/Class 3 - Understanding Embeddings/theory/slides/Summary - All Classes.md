# NLP Classes Summary: Converting Text to Numbers

## The Big Idea

**Computers only work with numbers. We have to convert text to numbers.**

**How we convert is super important** - different ways give us different results!

---

## 10 Key Points (What to Show on Slides)

### 1. **The Problem: Text → Numbers**
**Point to**: 
- A text example: "I love this movie"
- Show it needs to become numbers: `[0.2, 0.5, 0.1, 0.8, ...]`
- **Visual**: Text on left → Arrow → Numbers on right

**Toolkit**: `spaCy`, `NLTK` (for preprocessing before conversion)

### 2. **Old Way: TF-IDF (Sparse Vectors)**
**Point to**:
- Show sparse vector: `[0, 0, 0.8, 0, 0, 0.5, 0, 0, ...]` (mostly zeros!)
- Show "great" and "excellent" are in different positions
- **Visual**: Two words far apart in vector space

**Toolkit**: `scikit-learn` (TfidfVectorizer)

### 3. **New Way: Word Embeddings (Dense Vectors)**
**Point to**:
- Show dense vector: `[0.2, 0.5, 0.1, 0.8, 0.3, ...]` (all numbers matter!)
- Show "great" and "excellent" are close together
- **Visual**: t-SNE plot showing similar words clustering

**Toolkit**: `gensim` (Word2Vec, GloVe, FastText)

### 4. **Even Better: Sentence Embeddings**
**Point to**:
- Show sentence: "I love this movie" → one vector (not averaging words!)
- Show it captures word order and context
- **Visual**: Sentence-BERT vs word averaging comparison

**Toolkit**: `sentence-transformers` (Sentence-BERT)

### 5. **How They Learn: Adjust the Vectors**
**Point to**:
- Show three training methods side-by-side:
  - GPT: Predict next word → adjusts vectors
  - BERT: Predict missing word → adjusts vectors  
  - Sentence-BERT: Predict similarity → adjusts vectors
- **Visual**: Three diagrams showing "before" and "after" vectors

**Key concept**: **LEARN = Adjust the Vectors** (point to this!)

### 6. **Preprocessing: Clean Before Converting**
**Point to**:
- Show raw text: "I LOVE this movie!!!"
- Show cleaned: "i love this movie"
- **Visual**: Before/after text comparison

**Toolkit**: `spaCy`, `NLTK` (tokenization, cleaning)

### 7. **Visualization: See the Magic**
**Point to**:
- t-SNE plot: Similar words cluster together!
- Vector arithmetic: "king - man + woman = queen"
- **Visual**: Colorful t-SNE plot with word clusters

**Toolkit**: `matplotlib`, `seaborn` (for plotting)

### 8. **Sparse vs Dense: The Difference**
**Point to**:
- Sparse vector: Mostly zeros (show big vector with few non-zeros)
- Dense vector: All numbers matter (show compact vector)
- **Visual**: Side-by-side comparison (sparse = huge, dense = compact)

### 9. **What You Can Do: Applications**
**Point to**:
- TF-IDF → Keyword search (exact word matching)
- Word embeddings → Semantic search (meaning matching)
- Sentence embeddings → Classification (sentiment, categories)
- **Visual**: Three columns showing method → application

### 10. **The Progression**
**Point to**:
- Show the journey: TF-IDF → Word Embeddings → Sentence Embeddings
- Show each gets better at understanding meaning
- **Visual**: Flowchart or timeline showing progression

---

## Key Toolkits to Mention

**Point to toolkit logos/icons on slide:**

1. **`spaCy`** - Industry standard (tokenization, POS, NER, embeddings)
2. **`NLTK`** - Comprehensive NLP (preprocessing, tokenization)
3. **`gensim`** - Word embeddings (Word2Vec, GloVe, FastText)
4. **`sentence-transformers`** - Sentence embeddings (Sentence-BERT)
5. **`scikit-learn`** - Machine learning (TF-IDF, classification)

---

## The Bottom Line

**Point to this on your final slide:**

- **Computers need numbers** → We convert text to numbers
- **How we convert matters** → Better conversion = better understanding
- **Toolkits help** → `spaCy`, `NLTK`, `gensim`, `sentence-transformers`, `scikit-learn`

**The journey**: TF-IDF (word patterns) → Word Embeddings (word meaning) → Sentence Embeddings (sentence meaning)
