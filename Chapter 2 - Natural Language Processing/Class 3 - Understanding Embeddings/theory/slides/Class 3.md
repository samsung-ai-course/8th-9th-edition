# Class 3: Word Embeddings & Sentiment Analysis
## From Syntactic to Semantic Understanding

---

## Learning Objectives

By the end of this class, you will be able to:
- **EXPERIENCE** the difference between syntactic (TF-IDF) and semantic (embeddings) representations
- Understand how embeddings solve the limitations you learned about in Class 1 & 2
- Apply your geometric intuition (words as points in space) to semantic meaning (similar meanings = close points)
- Compare Word2Vec, GloVe, and FastText embeddings systematically
- Apply embeddings to sentiment analysis (supervised learning)
- Understand when preprocessing and linguistic features help vs hurt embeddings
- Use pre-trained embeddings effectively in real-world applications
- Make informed decisions: when to use TF-IDF vs embeddings

---

## Why This Class Matters: The Semantic Leap

### What You Already Know (Class 1 & 2)

**You've learned:**
- ✅ Text preprocessing (cleaning, tokenization, filtering)
- ✅ Bag of Words and TF-IDF (syntactic representations)
- ✅ Similarity search and clustering with TF-IDF
- ✅ **Geometric intuition**: Words as points in space (you already know this!)
- ✅ Distance metrics (cosine similarity, Euclidean distance)
- ✅ **The key limitation**: TF-IDF is **syntactic** (word-based, no meaning)
- ✅ TF-IDF treats synonyms as completely different ("great" ≠ "excellent")
- ✅ Evaluation metrics (Precision, Recall, F1-score)

**You understand the problem**: TF-IDF doesn't understand meaning! You know words as points in space, but in TF-IDF space, similar meanings are NOT close together.

### What This Class Adds: The Solution

**Today you'll learn:**
- 🎯 **Dense embeddings**: Semantic representations that understand meaning
- 🎯 **How embeddings solve** the synonym problem you learned about
- 🎯 **The key difference**: In embedding space, similar meanings ARE close together (unlike TF-IDF!)
- 🎯 **Practical application**: Sentiment analysis using embeddings
- 🎯 **When to use what**: TF-IDF vs embeddings decision framework

**The Key Insight:**
> **You already know the limitations. Today you'll see how embeddings solve them!**

---

## Recap: Known Limitations from Class 1 & 2

### The Syntactic Problem

**Remember from Class 2:**
- TF-IDF is **syntactic** (word-based, no meaning)
- **Semantic = meaning**
- TF-IDF treats synonyms as completely different words

**Concrete Example:**
```python
# These mean the same thing, but TF-IDF treats them as different:
"great movie"      # Vector 1
"excellent film"   # Vector 2

# TF-IDF similarity ≈ 0 (no shared words!)
# Humans know they're related, but TF-IDF doesn't!
```

**Visual Comparison:**
```
TF-IDF Space (Sparse):
"great"    [0, 0, 1, 0, 0, 0, ...]  ← Dimension 3
"excellent" [0, 0, 0, 0, 1, 0, ...]  ← Dimension 5
→ Distance: Far apart (different dimensions)

Embedding Space (Dense):
"great"    [0.2, -0.1, 0.5, 0.3, ...]  ← All dimensions matter
"excellent" [0.25, -0.08, 0.48, 0.32, ...]  ← Similar values!
→ Distance: Close together (similar vectors)
```

**The Analogy:**
> **TF-IDF is like a dictionary** - it knows exact words but not relationships
> 
> **Embeddings are like a thesaurus** - they understand related meanings

---

## Part 1: The Semantic Solution - Word Embeddings

### The Core Idea: Learning from Context

**The Big Idea:**
> "You shall know a word by the company it keeps" - J.R. Firth

**In Simple Terms:**
- Words that appear near each other often have similar meanings
- **Example**: If you always see "dog" near "bark", "leash", "pet", you learn what "dog" means
- **Like learning a language**: You learn words by seeing them in context

**Visual Example:**
```
Context: "I love _____"
Words filling the blank: "pizza", "coffee", "music", "movies"
→ These words are similar (all positive things)

Context: "The _____ is a programming language"
Words filling the blank: "Python", "Java", "JavaScript", "Ruby"
→ These words are similar (all programming languages)
```

**Why This Works:**
- No human labels needed! (That's why it's called "self-supervised")
- Learns from how words are used in real text
- Words in similar contexts → similar vectors → similar meanings

**Example in Code:**
```python
# Word2Vec learns from sentences like:
"I love pizza"
"I love coffee"  
"I love music"

# After training, "pizza", "coffee", "music" will have similar vectors
# because they all appear after "I love"
```

### Visual Understanding: Embedding Space

**Think of embeddings as a geometric map:**

```
                    Programming Languages
                        ↑
        Java ●      Python ●
                        |
                        |
    JavaScript ●--------|--------● Ruby
                        |
                        |
                    Languages

        Positive Sentiment
            ↑
    great ●    excellent ●
            |
            |
    amazing ●----|----● wonderful
            |
            |
        Positive Words
```

**In this space:**
- **Close words** = similar meanings
- **Directions** = relationships (king - man + woman = queen)
- **Distance** = semantic difference
- **Clusters** = related concepts

**Key Insight:**
> Embeddings create a **geometric representation of meaning** - words with similar meanings are close together in space!

---

## Word2Vec: Learning from Local Context

### Overview

**Word2Vec** (Mikolov et al., 2013) - Reference: [Original Paper](https://arxiv.org/abs/1301.3781)

**What it does:**
- Learns word embeddings from lots of text (millions of sentences)
- Two ways to train: **CBOW** and **Skip-gram**
- Self-supervised learning (no labels needed - learns from text itself!)

**In Python (using gensim):**
```python
import gensim.downloader as api

# Load pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Get word embedding
vector = model['python']  # Returns 300-dimensional vector

# Find similar words
similar = model.most_similar('python', topn=5)
# Returns: [('java', 0.72), ('javascript', 0.68), ...]
```

### Two Architectures: CBOW vs Skip-gram

#### 1. **CBOW (Continuous Bag of Words)**

**Concept**: Context → Predict center word

**Analogy**: "Fill in the blank"

```
"The quick brown _____ jumps over the lazy dog"
         ↓
    Predict: "fox"

Input: [the, quick, brown, jumps]
Output: fox
```

**Visualization:**
```
Context Words → [Neural Network] → Center Word
[the, quick, brown, jumps] → [NN] → "fox"
```

#### 2. **Skip-gram**

**Concept**: Center word → Predict context

**Analogy**: "Predict neighbors"

```
Center: "fox"
         ↓
Predict context: [the, quick, brown, jumps]

Input: fox
Output: [the, quick, brown, jumps]
```

**Visualization:**
```
Center Word → [Neural Network] → Context Words
"fox" → [NN] → [the, quick, brown, jumps]
```

**Which is Better?**
- **CBOW**: Faster, better for frequent words
- **Skip-gram**: Better for rare words, generally more accurate
- **In practice**: Skip-gram more commonly used

**Think Moment:**
> Why do you think Skip-gram works better for rare words?
> 
> *Hint: Rare words appear in fewer contexts, so predicting context from the word gives more training signal*

### How Word2Vec Trains (Simple Explanation)

**Step by Step:**
```python
"""
1. Start with random vectors for all words
   (Like random guesses - they'll get better!)

2. Slide a window through text:
   "The quick brown fox jumps over the lazy dog"
    [---- window ----]
    (Look at a few words at a time)

3. For each window:
   - CBOW: Given "the quick brown jumps", predict "fox"
   - Skip-gram: Given "fox", predict "the quick brown jumps"

4. If prediction is wrong, adjust the vectors
   (Move them closer together or farther apart)

5. Repeat millions of times
   (Go through entire Wikipedia, news articles, etc.)

6. Result: Words that appear together get similar vectors!
"""
```

**Real Example:**
```python
# After training on lots of text:
# "machine" and "learning" appear together often
# → Their vectors become similar

# "love" appears with "pizza", "coffee", "music"
# → "love" gets a positive sentiment vector
```

**Visual Example:**
```
Text: "I love machine learning"

Window 1: [I, love, machine] → predict "learning"
Window 2: [love, machine, learning] → predict "I"
Window 3: [machine, learning, ...] → predict "love"
...

After millions of windows:
- "machine" and "learning" appear together often
- They get similar vectors!
- "love" appears with positive words
- It gets a positive sentiment vector!
```

**The Magic:**
- No human labels needed!
- Learns from structure of language itself
- "You shall know a word by the company it keeps"

### Word2Vec Properties

#### 1. Semantic Similarity

**Example in Code:**
```python
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

# Find words similar to "great"
model.most_similar('great', topn=5)
# Returns: [('excellent', 0.75), ('amazing', 0.72), 
#           ('wonderful', 0.71), ('fantastic', 0.69)]

# Find words similar to "Python"
model.most_similar('Python', topn=5)
# Returns: [('Java', 0.68), ('JavaScript', 0.65), 
#           ('programming', 0.63), ('code', 0.61)]
```

**What this means:**
- "great" and "excellent" are close in embedding space (similar meaning!)
- "Python" and "Java" are close (both programming languages)

**Visualization:**
- t-SNE plot showing "great" and "excellent" close together
- "Python" and "Java" in same cluster
- Positive words cluster together
- Negative words cluster together

#### 2. Vector Arithmetic (The Famous Example)

**The Cool Trick:**
```python
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

# Vector arithmetic!
result = model.most_similar(
    positive=['woman', 'king'], 
    negative=['man'], 
    topn=1
)
# Returns: [('queen', 0.71)]

# In math terms:
# vector(king) - vector(man) + vector(woman) ≈ vector(queen)
```

**What this shows:**
- Embeddings capture relationships, not just similarity
- The "royalty" relationship is preserved in the vectors

**Geometric Visualization:**
```
        king
         ●
         |
    man  |  woman
     ●   |   ●
     \   |   /
      \  |  /
       \ | /
        \|/
         ●
       queen

The relationship "royalty - gender + other gender" is preserved!
```

**More Examples:**
```python
# Capital cities
model.most_similar(
    positive=['Italy', 'Paris'], 
    negative=['France']
)
# Returns: [('Rome', 0.68)]  # Capital of Italy!

# Word relationships
model.most_similar(
    positive=['music', 'Python'], 
    negative=['programming']
)
# Returns: [('song', 0.65)]  # Python → programming, music → song
```

**Key Insight:**
> Embeddings capture **relationships**, not just similarity!
> 
> Directions in embedding space represent semantic relationships

#### 3. Compositionality

**Vectors can be combined:**
```python
"New" + "York" ≈ "New York"
"machine" + "learning" ≈ "machine learning"
```

---

## GloVe: Global Vectors

### Overview

**GloVe** (Pennington et al., 2014) - Reference: [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)

**What it does:**
- Looks at ALL word pairs in the entire text (not just nearby words)
- Counts how often words appear together
- Often works better than Word2Vec

**In Python:**
```python
import gensim.downloader as api

# Load pre-trained GloVe model
model = api.load('glove-wiki-gigaword-300')

# Same interface as Word2Vec!
vector = model['python']
similar = model.most_similar('python', topn=5)
```

### Key Difference from Word2Vec

**Word2Vec:**
- Looks at nearby words only (sliding window)
- Example: "The cat sat on the mat" → looks at 5 words at a time
- **Analogy**: "Local gossip" - learns from immediate neighbors

**GloVe:**
- Looks at ALL word pairs in entire text
- Counts: "How many times did 'cat' and 'sat' appear together in ALL text?"
- **Analogy**: "Global reputation" - learns from overall patterns

**Example:**
```python
# Word2Vec: "The cat sat" → learns from this window
# GloVe: Counts "cat"+"sat" across entire Wikipedia
#        If they appear together 1000 times → similar vectors
```

### How GloVe Works (High-Level)

**The Process:**
```python
"""
1. Build co-occurrence matrix:
   Count how often words appear together across entire corpus

   Example:
          | the | cat | sat | mat |
   -------|-----|-----|-----|-----|
   the    |  0  | 100 |  50 |  30 |
   cat    | 100 |  0  |  80 |  10 |
   sat    |  50 |  80 |  0  |  70 |
   mat    |  30 |  10 |  70 |  0  |

2. Factorize matrix into word vectors
3. Words that co-occur frequently → similar vectors
"""
```

**Visualization:**
- **Co-occurrence matrix heatmap**: Shows word pairs that appear together
- Darker colors = more frequent co-occurrence
- Words with similar co-occurrence patterns → similar vectors

**Example Co-occurrence Pattern:**
```
"cat" co-occurs with: "sat" (80 times), "the" (100 times), "mat" (10 times)
"dog" co-occurs with: "sat" (75 times), "the" (95 times), "mat" (12 times)
→ "cat" and "dog" have similar co-occurrence patterns
→ They get similar vectors!
```

### GloVe vs Word2Vec Comparison

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| **Training** | Local context (sliding window) | Global statistics (entire corpus) |
| **Speed** | Faster on small data | Faster on large data |
| **Accuracy** | Good | Often better |
| **Memory** | Less memory | Needs more memory (co-occurrence matrix) |
| **Use case** | General purpose | When you have large corpus |
| **Analogy** | Local gossip | Global reputation |

**Think Moment:**
> When would global statistics help vs local context?
> 
> *Hint: Think about rare words vs common words, domain-specific vs general text*

**In Practice**: Both work well! Try both and see which works better for your data.

---

## FastText: Handling the Unknown

### Overview

**FastText** (Bojanowski et al., 2017) - Reference: [FastText Paper](https://arxiv.org/abs/1607.04606)

**What it does:**
- Handles unknown words better (OOV = Out Of Vocabulary)
- Breaks words into smaller pieces (subwords)
- Better for typos, rare words, and languages with lots of word forms

**In Python:**
```python
import gensim.downloader as api

# Load FastText model
model = api.load('fasttext-wiki-news-subwords-300')

# Works even with typos!
vector = model['teh']  # Typo of "the" - still works!
similar = model.most_similar('teh', topn=5)
```

### The OOV Problem

**Problem with Word2Vec/GloVe:**
```python
# If word not in training data:
word2vec["unhappiness"]  # KeyError! Word not found
glove["unhappiness"]     # KeyError! Word not found

# Even if you know:
word2vec["happy"]        # Works
word2vec["un"]           # Might work
```

**FastText Solution:**
```python
# FastText breaks words into subwords (character pieces):
"unhappiness" → ["un", "unh", "hap", "app", "ppi", "pin", "ine", "nes", "ess"]

# Gets embedding by combining subword embeddings:
fasttext["unhappiness"] = average(subword_embeddings)
# Works even if "unhappiness" was never seen!

# Example:
from gensim.models import FastText
# Even if model never saw "misunderstood":
vector = model["misunderstood"]  # Works! Uses subwords: "mis", "under", "stood"
```

### Subword Information

**How It Works:**
- Breaks words into character n-grams (3-6 characters)
- Learns embeddings for subwords
- Word embedding = average of subword embeddings

**Visual Example:**
```
Word: "unhappiness"
Subwords (n=3): ["un", "unh", "hap", "app", "ppi", "pin", "ine", "nes", "ess"]
Subwords (n=4): ["unha", "nhap", "happ", "appi", "ppin", "pine", "ines", "ness"]

Final embedding = weighted average of all subword embeddings
```

**Analogy:**
> Like recognizing a word even if it's misspelled - you recognize parts!
> 
> Or like syllables - you understand "un-happy-ness" from its parts

### OOV Handling Comparison

**Example:**
```python
# Unknown word: "misunderstood"

Word2Vec:  KeyError (word not in vocabulary)
GloVe:     KeyError (word not in vocabulary)
FastText:  Works! Uses subwords: ["mis", "und", "der", "stoo", "ood"]
```

**Visualization:**
- Show OOV handling comparison chart
- FastText handles typos, rare words, morphological variations

**Think Moment:**
> Why would subwords help with typos and rare words?
> 
> *Hint: Even if you've never seen "misunderstood", you've probably seen "mis", "under", "stood"*

### When to Use FastText

**Use FastText when:**
- ✅ Dealing with typos and misspellings
- ✅ Rare words are important
- ✅ Morphologically rich languages
- ✅ Social media text (lots of OOV words)
- ✅ Domain-specific vocabulary

**Use Word2Vec/GloVe when:**
- ✅ Clean, well-edited text
- ✅ Common vocabulary
- ✅ Speed is critical (FastText is slower)

---

## From Word Embeddings to Sentences & Documents

### The Question: "OK, We Have Word Embeddings, But What About Sentences?"

**You might be thinking:**
> "Great! We have embeddings for words. But most tasks need sentence or document embeddings. How do we get from word → sentence → document?"

**The Answer:** There are several approaches, from simple to sophisticated!

### Approach 1: Simple Averaging (What You're Doing Now)

**The Simplest Method:**
```python
import numpy as np
from gensim.models import Word2Vec

# Sentence: "I love machine learning"
sentence = "I love machine learning"
words = sentence.split()  # ["I", "love", "machine", "learning"]

# Get word embeddings
model = Word2Vec.load('word2vec.model')
word_vectors = [model[word] for word in words if word in model]

# Simple average:
sentence_embedding = np.mean(word_vectors, axis=0)
# = (vector("I") + vector("love") + vector("machine") + vector("learning")) / 4
```

**In the Notebook:**
```python
def average_word_vectors(text, model, dim=300):
    """Create document embedding by averaging word vectors"""
    words = text.lower().split()
    vectors = []
    
    for word in words:
        word = word.strip('.,!?;:()[]{}"\'').lower()
        if word in model:
            vectors.append(model[word])
    
    if len(vectors) == 0:
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)

# Use it:
doc_embedding = average_word_vectors("I love machine learning", model)
```

**Visualization:**
```
Word Embeddings:
"I"      → [0.1, 0.2, 0.3, ...]  ●
"love"   → [0.5, 0.3, 0.1, ...]  ●
"machine"→ [0.2, 0.4, 0.2, ...]  ●
"learning"→ [0.3, 0.1, 0.4, ...]  ●

Sentence Embedding (average):
[0.275, 0.25, 0.25, ...]  ● (centroid of word vectors)
```

**Pros:**
- ✅ Simple and fast
- ✅ Works reasonably well
- ✅ No additional training needed
- ✅ What you're using in Notebook Part 1!

**Cons:**
- ❌ Loses word order (same as BoW!)
- ❌ All words weighted equally (even "the", "a", "is")
- ❌ Doesn't capture sentence structure

**Analogy:**
> Like averaging all colors in a painting - you get the overall tone, but lose the composition

**This is what you're doing in the notebooks!** It works, but there are better ways.

### Approach 2: Weighted Averaging (TF-IDF Weighting)

**Better: Weight Important Words More:**
```python
# Weight by TF-IDF scores:
sentence = "I love machine learning"
tfidf_scores = {"I": 0.1, "love": 0.8, "machine": 0.9, "learning": 0.9}

# Weighted average:
sentence_embedding = weighted_mean(word_embeddings, weights=tfidf_scores)
# "love", "machine", "learning" contribute more than "I"
```

**Visualization:**
```
Word Embeddings (size = importance):
"I"      → [0.1, 0.2, ...]  ● (small - low weight)
"love"   → [0.5, 0.3, ...]  ●●● (medium - medium weight)
"machine"→ [0.2, 0.4, ...]  ●●●● (large - high weight)
"learning"→ [0.3, 0.1, ...]  ●●●● (large - high weight)

Sentence Embedding (weighted centroid):
Closer to "machine" and "learning" (important words)
```

**Pros:**
- ✅ Better than simple averaging
- ✅ Important words contribute more
- ✅ Still simple to implement

**Cons:**
- ❌ Still loses word order
- ❌ Requires TF-IDF computation
- ❌ Not as good as specialized methods

**When to Use:**
- Quick improvements over simple averaging
- When you have TF-IDF scores already
- Good baseline before trying advanced methods

### Approach 3: Sentence Embeddings (Sentence-BERT, Universal Sentence Encoder)

**The Best Approach: Specialized Sentence Embeddings**

**What They Are:**
- Models trained specifically to create sentence embeddings
- Use neural networks to combine word embeddings (not just averaging!)
- Trained on sentence pairs (contrastive learning - we'll learn about this!)

**How They're Built:**
- **Sentence-BERT**: Built on top of BERT (which uses Masked Language Modeling - see Training Objectives section!)
  - BERT learns word embeddings from bidirectional context (MLM)
  - Sentence-BERT fine-tunes BERT with contrastive learning for sentence similarity
  - **Two-stage training**: First MLM (word understanding), then contrastive (sentence similarity)
- **Other models**: May use different base models (GPT-style, BERT-style, or custom architectures)

**How They Work:**
```python
from sentence_transformers import SentenceTransformer

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentence (much better than averaging!)
sentence = "I love machine learning"
sentence_embedding = model.encode(sentence)
# Returns: numpy array of shape (384,) - sentence embedding!

# Compare sentences
sentences = [
    "I love machine learning",
    "I hate machine learning",
    "Machine learning is great"
]
embeddings = model.encode(sentences)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[2]])
# Returns: High similarity (both positive about ML)
```

**Reference:** [Sentence Transformers Library](https://www.sbert.net/)

**Visualization:**
```
Word Embeddings:
"I"      → [0.1, 0.2, ...]  ●
"love"   → [0.5, 0.3, ...]  ●
"machine"→ [0.2, 0.4, ...]  ●
"learning"→ [0.3, 0.1, ...]  ●

↓ Neural Network (learns how to combine) ↓

Sentence Embedding:
[0.4, 0.3, 0.5, ...]  ● (captures sentence meaning!)
```

**Key Advantages:**
- ✅ Captures sentence-level meaning
- ✅ Better than averaging (trained for this!)
- ✅ Handles word order implicitly
- ✅ Works great for similarity tasks

**Examples:**
- **Sentence-BERT**: Built on BERT (MLM) + fine-tuned with contrastive learning
- **Universal Sentence Encoder**: Google's sentence embedding model (uses transformer architecture)
- **InferSent**: Facebook's sentence embedding model

**Connection to Training Objectives:**
> Sentence-BERT combines two training objectives:
> 1. **Masked Language Modeling** (from BERT) - learns word understanding from context
> 2. **Contrastive Learning** (fine-tuning) - learns sentence similarity
> 
> This is why it works better than averaging! It has both word-level understanding (from MLM) and sentence-level training (from contrastive learning).
> 
> See the "Training Objectives" section below for details on MLM and contrastive learning!

**You'll see this in Notebook Part 2!**

### Comparison: Which Approach to Use?

**Decision Framework:**

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Simple Averaging** | Quick prototyping, baseline | Fast, simple | Loses word order, equal weights |
| **Weighted Averaging** | Better than simple, have TF-IDF | Better than simple | Still loses word order |
| **Sentence Embeddings** | Best results, similarity tasks | Best quality | Requires model download |

**Visual Comparison:**
```
Task: Find similar sentences

Simple Average:
"I love Python" vs "Python I love" → Similar (same words)
"I love Python" vs "I hate Python" → Similar (most words same) ❌

Sentence-BERT:
"I love Python" vs "Python I love" → Similar ✅
"I love Python" vs "I hate Python" → Different ✅
```

### The Key Insight

**For Documents (Longer Text):**
- Same approaches work!
- Average word embeddings → document embedding
- Or use sentence embeddings for each sentence, then average sentences
- Or use specialized document embedding models

**What You're Learning:**
1. **Word embeddings** → Understand individual words
2. **Sentence embeddings** → Understand sentences (combine words intelligently)
3. **Document embeddings** → Understand documents (combine sentences)

**In This Class:**
- **Part 1**: Word embeddings (Word2Vec, GloVe, FastText) + simple averaging
- **Part 2**: Sentence-BERT (specialized sentence embeddings) + comparison

**The Progression:**
```
Words → Sentences → Documents
  ↓         ↓          ↓
Word2Vec  Averaging  Averaging
GloVe     Weighted   Weighted
FastText  Sentence-  Sentence-
          BERT       BERT
```

**Think Moment:**
> Why would sentence embeddings be better than averaging word embeddings?
> 
> *Hint: They're trained specifically for sentence-level tasks, not just word-level tasks*

---

## Training Objectives: How Embeddings Learn

### Understanding How Embeddings Are Trained

**Now that you know what embeddings are and how to use them, let's see HOW they're trained.**

**Why this matters:**
- Explains why sentence embeddings work better than averaging
- Shows how different models (GPT, BERT, Sentence-BERT) differ
- Helps you understand what makes embeddings capture meaning

**The Big Picture:**
- **Word2Vec/GloVe**: Train on word-level tasks (predict nearby words, count co-occurrences)
- **GPT/BERT**: Train on sentence-level tasks (understand sequences/context - better sentence understanding than Word2Vec/GloVe)
- **Sentence-BERT**: Train specifically for sentence similarity (best for sentence embeddings and similarity tasks!)

Embeddings can be trained using different objectives. Understanding these helps you understand modern NLP!

### 1. Next Token Prediction (GPT-style)

**What it does**: Predict the next word in a sequence

**Example:**
```
Input: "The cat sat on"
Output: predict "the"

Input: "The cat sat on the"
Output: predict "mat"
```

**How it works:**
- Reads left-to-right (like reading a book)
- Given words so far, predict what comes next
- Trained on huge amounts of text (Wikipedia, books, web)

**Analogy:**
> Like autocomplete on your phone - predicts what you'll type next

**What it powers**: GPT models (GPT-2, GPT-3, GPT-4, ChatGPT)

**Try it yourself:**
- Demo: [HuggingFace Text Generation](https://huggingface.co/spaces)
- Code: `from transformers import pipeline; generator = pipeline('text-generation')`

**Key Point:**
- Learns language patterns (what words usually come after what)
- Can generate new text
- Only sees left context (not right)

### 2. Masked Language Modeling (BERT-style)

**What it does**: Predict masked words in context

**Example:**
```
Input: "The cat [MASK] on the mat"
Output: predict "sat" (most likely word)

Input: "The [MASK] sat on the mat"
Output: predict "cat" (most likely word)
```

**How it works:**
- Sees BOTH left AND right context (unlike GPT!)
- Randomly hides words (masks them)
- Predicts what the hidden word should be

**Analogy:**
> Like fill-in-the-blank - you see words before AND after the blank

**What it powers**: BERT models (BERT, RoBERTa, DistilBERT)

**Try it yourself:**
- Demo: [HuggingFace Fill-Mask](https://huggingface.co/spaces)
- Code: `from transformers import pipeline; fill_mask = pipeline('fill-mask')`

**Key Point:**
- Uses bidirectional context (sees both sides - better than GPT!)
- Better for understanding text (not generating)
- More context-aware than GPT

### 3. Contrastive Learning (Sentence Embeddings)

**What it does**: Similar sentences close together, dissimilar sentences far apart

**Example:**
```
Embedding Space:

Similar sentences (close):
"I love this movie" ●
"This film is great" ●  ← Close together (similar meaning!)

Dissimilar sentences (far):
"I love this movie" ●
"I hate this movie" ●  ← Far apart (opposite meaning!)
```

**How it works:**
- Train on sentence pairs (similar/dissimilar)
- Similar pairs → move vectors closer together
- Dissimilar pairs → move vectors farther apart

**Example in Code:**
```python
# Training data:
similar_pairs = [
    ("I love this movie", "This film is great"),  # Similar!
    ("The weather is nice", "It's sunny today")   # Similar!
]

dissimilar_pairs = [
    ("I love this movie", "I hate this movie"),   # Different!
    ("The weather is nice", "It's raining")       # Different!
]

# Model learns: similar sentences → close vectors
```

**Analogy:**
> Like organizing a library - similar books go on same shelf

**What it powers**: Sentence-BERT, Universal Sentence Encoder

**Key Point:**
- Trained specifically for sentence similarity (not word similarity!)
- Much better than averaging word embeddings
- Captures sentence-level meaning

**Visual Example:**
- t-SNE plot showing similar sentences clustering together
- Positive reviews cluster, negative reviews cluster
- Related topics cluster

**Connection to Sentence Embeddings:**
> This is why Sentence-BERT works better than averaging! It's trained specifically for sentence similarity using contrastive learning.

---

## Neural Networks: How Embeddings Are Learned

### The Big Idea

**You know embeddings are trained, but what does the training? Neural networks!**

**Key point**: Neural networks are the tools that learn embeddings. They adjust vectors based on training objectives.

### What Are Neural Networks?

**Simple idea**: Think of a neural network like a student learning patterns:
- Sees examples (words in context, sentences)
- Notices patterns ("these words appear together often")
- Creates embeddings to represent what it learned

**What they are**:
- Computational models inspired by how brains work
- Made of interconnected "neurons" (simple processing units)
- Learn by adjusting connections based on examples
- For embeddings: They adjust vector values to capture patterns in text

**Important**: You don't need to understand neural network details yet! This is just the big picture. You'll learn the details in deep learning chapters.

### How Neural Networks Learn Embeddings

**The process**:

1. **Start with random vectors**
   - Each word gets a random vector (like random guesses)
   - These improve through training

2. **Feed text to the network**
   - Network sees millions of sentences
   - For each sentence, tries to predict something (next word, masked word, similarity)

3. **Make predictions**
   - Network uses current vectors to predict
   - Compares predictions to actual answers

4. **Adjust vectors if wrong**
   - If prediction is wrong, network adjusts vectors
   - Moves vectors closer together or farther apart
   - This is the learning part!

5. **Repeat millions of times**
   - Process repeats for entire training corpus
   - Vectors gradually improve
   - Eventually capture semantic relationships

**Visual Analogy**:
```
Initial State (Random):
"great"    [0.1, -0.3, 0.5, ...]  ← Random numbers
"excellent" [0.4, 0.2, -0.1, ...]  ← Random numbers
→ Distance: Far apart (random)

After Training (Learned):
"great"    [0.2, -0.1, 0.5, 0.3, ...]  ← Adjusted numbers
"excellent" [0.25, -0.08, 0.48, 0.32, ...]  ← Adjusted numbers
→ Distance: Close together (learned similarity!)
```

**Key idea**: **LEARN = Adjust the Vectors**
- Learning means changing the numbers in embeddings
- Neural networks do this automatically based on training objectives
- Different objectives (next-token, MLM, contrastive) adjust vectors differently

### Connection to Training Objectives

**Remember the three training objectives? Neural networks execute them:**

1. **Next Token Prediction (GPT-style)**
   - Network sees: "The cat sat on"
   - Predicts: "the" (next word)
   - If wrong → adjusts vectors
   - Result: Vectors learn language patterns

2. **Masked Language Modeling (BERT-style)**
   - Network sees: "The cat [MASK] on the mat"
   - Predicts: "sat" (masked word)
   - Uses bidirectional context (both left and right words)
   - If wrong → adjusts vectors
   - Result: Vectors learn contextual meaning

3. **Contrastive Learning (Sentence-BERT-style)**
   - Network sees: Similar sentence pairs, dissimilar pairs
   - Learns: Push similar sentences closer, dissimilar farther
   - Adjusts vectors for better sentence representations
   - Result: Vectors optimized for sentence similarity

**Pattern**: All three use neural networks to adjust vectors, but with different goals!

### Why Neural Networks Work for Embeddings

**What makes them powerful**:

1. **Pattern recognition**
   - Networks excel at finding patterns in data
   - Text has patterns: words in similar contexts have similar meanings
   - Networks learn these patterns automatically

2. **Automatic learning**
   - No human needs to define rules
   - Network discovers relationships from data
   - Learns complex patterns humans might miss

3. **Scalability**
   - Can process millions/billions of examples
   - Learns from massive text corpora
   - Gets better with more data

4. **Flexibility**
   - Same architecture can learn different objectives
   - Can adapt to different tasks (word-level, sentence-level)
   - Can be fine-tuned for specific domains

### Different Models, Different Architectures

**Word2Vec/GloVe**:
- Simpler neural network architectures
- Focus on word-level patterns
- Fast training, good for common words
- Architecture: Feedforward networks (you'll learn details later!)

**BERT/GPT**:
- More complex architectures (Transformers)
- Better at understanding context and sequences
- Can handle longer text, better sentence understanding
- Architecture: Transformer networks (advanced topic for later!)

**Sentence-BERT**:
- Built on top of BERT (uses Transformer architecture)
- Fine-tuned with contrastive learning
- Optimized for sentence similarity
- Architecture: Transformer + contrastive training

**Key point**: More complex architectures (like Transformers) learn better representations, but they're also more expensive. You'll learn about these in deep learning chapters!

### The Big Picture: From Text to Meaning

**The journey**:

```
Raw Text
    ↓
Neural Network (sees patterns)
    ↓
Training Objective (what to predict)
    ↓
Vector Adjustments (learning)
    ↓
Embeddings (learned representations)
    ↓
Semantic Understanding (meaning captured!)
```

**What you need to know now**:
- ✅ Neural networks learn embeddings
- ✅ They adjust vectors based on training objectives
- ✅ Different objectives create different types of embeddings
- ✅ More complex networks learn better representations
- ❌ You don't need architecture details yet!

**What you'll learn later** (Deep Learning Chapters):
- How neural networks work (neurons, layers, activation functions)
- Different architectures (feedforward, recurrent, transformer)
- How training happens (backpropagation, optimization)
- How to build and train your own networks

### Practical Implications

**For using embeddings**:
- You don't need to understand neural networks to use embeddings!
- Pre-trained embeddings already learned from neural networks
- Just load and use them (like we've been doing)

**For understanding embeddings**:
- Knowing neural networks learn them helps explain:
  - Why embeddings capture meaning (learned from patterns)
  - Why different models give different results (different architectures/objectives)
  - Why training on more data helps (more patterns to learn)

**For future learning**:
- Understanding embeddings prepares you for neural networks
- You've seen what neural networks can do (create semantic representations)
- You'll learn how they do it (architecture, training) in deep learning chapters

### Key Takeaways

1. **Neural networks are the learning machines** that create embeddings
2. **They adjust vectors** based on training objectives (next-token, MLM, contrastive)
3. **Different architectures** (simple → complex) learn different quality embeddings
4. **You don't need details yet** - just understand the big picture!
5. **Pre-trained embeddings** already did the neural network training for you

**Remember**: This is an introduction! You'll dive deep into neural networks in later chapters. For now, just understand that they're the tools that learn embeddings from text.

---

## Pre-trained Embeddings: Standing on Giants' Shoulders

### Why Pre-trained?

**Training your own embeddings requires:**
- Massive text corpus (millions/billions of words)
- Significant computational resources (GPUs, days/weeks)
- Expertise in hyperparameter tuning

**Pre-trained embeddings:**
- ✅ Already trained on huge datasets (Wikipedia, Common Crawl, Google News)
- ✅ Ready to use immediately
- ✅ Generally better than what you can train yourself
- ✅ Free and open-source

### Popular Pre-trained Options

#### 1. **Word2Vec (Google)**
- Trained on Google News (100 billion words)
- 300 dimensions
- 3 million words
- **Best for**: General purpose, fast

#### 2. **GloVe (Stanford)**
- Multiple versions:
  - glove.6B: 6 billion tokens (Wikipedia + Gigaword)
  - glove.42B: 42 billion tokens (Common Crawl)
  - glove.840B: 840 billion tokens (Common Crawl)
- Dimensions: 50, 100, 200, 300
- **Best for**: Large-scale applications, often better accuracy

#### 3. **FastText (Facebook)**
- Handles OOV words better
- Uses subword information
- **Best for**: Social media, typos, rare words

#### 4. **spaCy Embeddings**
- Built-in with spaCy models
- Easy to use, integrated
- **Best for**: Quick prototyping, production systems

#### 5. **Sentence-BERT**
- Pre-trained sentence embeddings
- Uses contrastive learning (as we learned!)
- **Best for**: Sentence similarity, semantic search

### When to Train Your Own

**Train your own if:**
- Domain-specific vocabulary (medical, legal, technical)
- Pre-trained don't perform well on your data
- You have massive domain-specific corpus
- You have computational resources

**Use pre-trained if:**
- General domain
- Limited data
- Limited resources
- Need to start quickly

**Hybrid Approach** (Best of both worlds):
```python
# Start with pre-trained embeddings
# Fine-tune on your domain-specific data
# → Better results than either alone!
```

---

## Linguistic Features: When They Help

### POS Tagging: Filtering Sentiment-Bearing Words

**Concept**: Part-of-speech tags (noun, verb, adjective, adverb)

**Why It Matters for Sentiment:**
- Adjectives and adverbs often carry sentiment
- Example: "absolutely terrible" (ADV + ADJ)
- Nouns and verbs less likely to carry sentiment

**Visual Example:**
```
Sentence: "This movie is absolutely terrible"
POS tags:
- "This" → DET (determiner)
- "movie" → NOUN
- "is" → VERB
- "absolutely" → ADV (adverb) ← Sentiment-bearing!
- "terrible" → ADJ (adjective) ← Sentiment-bearing!
```

**When It Helps:**
- ✅ Filtering to ADJ + ADV only can improve sentiment analysis
- ✅ Removes noise from other parts of speech
- ✅ Focuses on words that matter

**When It Doesn't Help:**
- ❌ Sometimes context matters (verbs can carry sentiment: "hate", "love")
- ❌ Nouns can be important ("disaster", "masterpiece")
- ❌ May lose important information

**Think Moment:**
> Do sentiment-bearing words alone work better than full text?
> 
> *Try it and see! Sometimes less is more, sometimes context matters*

### Named Entity Recognition (NER)

**Concept**: Identify named entities (PERSON, ORG, LOCATION, PRODUCT)

**When It Matters for Sentiment:**
- Sometimes entities distract from sentiment
- Example: "Apple products" vs "apple taste"
  - "Apple" (company) vs "apple" (fruit) - different sentiment contexts

**Visual Example:**
```
Sentence: "I love Apple products but hate the price"
Entities:
- "Apple" → ORG (organization)
- "products" → (not an entity)
- "price" → (not an entity)

Options:
1. Remove entities: "I love products but hate the price" (loses context)
2. Replace with [ENTITY]: "I love [ORG] products but hate the price" (preserves structure)
3. Keep as-is: "I love Apple products but hate the price" (full context)
```

**When It Helps:**
- ✅ When entities are distracting (product names, company names)
- ✅ When you want to generalize (not specific to one entity)
- ✅ When entities don't carry sentiment

**When It Doesn't Help:**
- ❌ When entities are important for sentiment ("I love Apple" - entity IS the sentiment)
- ❌ When removing entities loses meaning
- ❌ When entities provide context

**Think Moment:**
> When does NER help vs hurt for sentiment analysis?
> 
> *It depends on your use case!*

---

## The Complete Pipeline: Sentiment Analysis

### From Text to Prediction

**Full Pipeline:**
```
Raw Text
    ↓
Preprocessing (optional - sometimes less is more!)
    ↓
Tokenization
    ↓
Word Embeddings (Word2Vec/GloVe/FastText)
    ↓
Document Embedding (average word vectors)
    ↓
Classifier (Logistic Regression, etc.)
    ↓
Prediction (positive/negative)
```

### Preprocessing Impact

**Key Insight**: Preprocessing affects embeddings differently than TF-IDF!

**With TF-IDF:**
- More preprocessing usually helps (remove noise, normalize)
- Stop words removal helps
- Lemmatization helps

**With Embeddings:**
- Sometimes less preprocessing is better!
- Stop words can provide context
- Over-cleaning can lose information
- **Negation handling is critical**: "not good" ≠ "good"

**Visualization:**
- Comparison table: preprocessing strategies vs accuracy
- Bar chart showing impact
- Example: "not good" vs "good" - how negation changes meaning

**Think Moment:**
> Why might less preprocessing help embeddings?
> 
> *Hint: Embeddings learn from context - removing too much loses context*

### Evaluation: Comparing Approaches

**Metrics:**
- Accuracy: Overall correctness
- Precision: When we predict positive, how often are we right?
- Recall: How many actual positives did we catch?
- F1-Score: Balance between precision and recall
- Confusion Matrix: Visual representation of errors

**Visualization:**
- Confusion matrix heatmaps for each approach
- Comparison bar charts
- t-SNE plots showing how different approaches cluster

---

## Trade-offs: When to Use What

### Decision Framework

| Factor | Use TF-IDF (Sparse) | Use Embeddings (Dense) |
|--------|-------------------|----------------------|
| **Semantic understanding** | Not needed | Critical |
| **Synonym handling** | Not important | Important |
| **Dataset size** | Small (<1000 docs) | Any size |
| **Speed requirements** | Critical (very fast) | Moderate |
| **Interpretability** | Very important | Less important |
| **Domain** | General/stable | Specialized/evolving |
| **Resources** | Limited | Available |
| **OOV words** | Not a problem | FastText handles better |

### Hybrid Approach (Best of Both Worlds!)

**Combine TF-IDF and embeddings:**
```python
def hybrid_search(query, documents, alpha=0.5):
    # TF-IDF scores (exact matching)
    tfidf_scores = tfidf_similarity(query, documents)
    
    # Embedding scores (semantic matching)
    embedding_scores = embedding_similarity(query, documents)
    
    # Combine (weighted average)
    final_scores = alpha * tfidf_scores + (1 - alpha) * embedding_scores
    
    return ranked_results(final_scores)

# alpha=0.5: Equal weight to both
# alpha=0.7: More weight to TF-IDF (for exact matching)
# alpha=0.3: More weight to embeddings (for semantic matching)
```

**When to use hybrid:**
- Legal/medical search (need both exact terms AND semantics)
- E-commerce (exact product names AND similar products)
- Academic search (exact citations AND related papers)

---

## Summary: Key Takeaways

### The Semantic Leap

1. ✅ **You already knew**: TF-IDF is syntactic (no meaning)
2. ✅ **Today you learned**: Embeddings are semantic (understand meaning)
3. ✅ **You experienced**: How embeddings solve the synonym problem
4. ✅ **You built intuition**: Geometric understanding of vector spaces

### Embedding Methods

5. ✅ **Word2Vec**: Learns from local context (CBOW, Skip-gram)
6. ✅ **GloVe**: Uses global co-occurrence statistics
7. ✅ **FastText**: Handles OOV words with subwords
8. ✅ **Pre-trained**: Use them! Don't train from scratch

### Practical Mastery

9. ✅ **Preprocessing**: Affects embeddings differently than TF-IDF (sometimes less is more!)
10. ✅ **Linguistic features**: POS and NER can help, but not always
11. ✅ **Evaluation**: Compare approaches systematically
12. ✅ **Decision framework**: Know when to use TF-IDF vs embeddings

### The Big Picture

```
Good Preprocessing + Good Representations = Good Results

TF-IDF: Fast, interpretable, but no semantics (syntactic)
Embeddings: Semantic understanding, slower, less interpretable
Hybrid: Best of both worlds!
```

---

## Practice Tips

### For The Notebooks

1. **Experience the difference**:
   - See how "great" and "excellent" are close in embedding space (unlike TF-IDF!)
   - Compare TF-IDF vs embeddings side-by-side
   - Apply your geometric intuition to semantic relationships

2. **Experiment with preprocessing**:
   - Try different strategies
   - See impact on results
   - Understand when less is more

3. **Compare methods**:
   - TF-IDF vs Word2Vec vs GloVe vs FastText
   - Simple average vs weighted
   - Word averaging vs sentence embeddings

4. **Visualize**:
   - Use t-SNE to see semantic clusters
   - Compare confusion matrices
   - Understand relationships geometrically

5. **Think critically**:
   - When does preprocessing help vs hurt?
   - When do linguistic features matter?
   - Which approach works best for your use case?

---

## Ready for The Exercises?

✅ You understand how embeddings solve Class 1 & 2 limitations  
✅ You have geometric intuition about vector spaces  
✅ You know the differences between Word2Vec, GloVe, and FastText  
✅ You understand when preprocessing helps vs hurts  
✅ You can use pre-trained embeddings effectively  
✅ You know when to use TF-IDF vs embeddings  

**Let's apply this to sentiment analysis in the notebooks!** 🚀

---

## Resources

### Documentation
- [Gensim](https://radimrehurek.com/gensim/) - Word2Vec, GloVe, FastText implementations
- [spaCy](https://spacy.io) - Industrial-strength NLP with embeddings
- [Sentence Transformers](https://www.sbert.net/) - Sentence embeddings

### Pre-trained Models
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- [Word2Vec Models](https://code.google.com/archive/p/word2vec/)
- [FastText Models](https://fasttext.cc/)
- [spaCy Models](https://spacy.io/models)

### Visualization Tools
- [Embedding Projector](https://projector.tensorflow.org/) - Visualize embeddings interactively
- [t-SNE Visualization](https://distill.pub/2016/misread-tsne/) - Understanding t-SNE

### Demos
- [HuggingFace Fill-Mask](https://huggingface.co/spaces) - Try BERT-style models
- [HuggingFace Text Generation](https://huggingface.co/spaces) - Try GPT-style models
