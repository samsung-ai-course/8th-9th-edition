# Class 1 & 2: FAQ - Common Questions & Answers

## Text Preprocessing

### Q: What's the difference between pre-processing, tokenization, and post-processing?
**A**: They happen in order:
- **Pre-processing**: Clean raw text (remove URLs, emails, HTML)
- **Tokenization**: Split text into words/tokens
- **Post-processing**: Refine tokens (remove stop words, filter)

**Example:**
```
Raw: "Contact: john@email.com for info!"
Pre-process: "Contact:  for info!" (removed email)
Tokenize: ["Contact", "for", "info"]
Post-process: ["Contact", "info"] (removed "for" - stop word)
```

---

### Q: I don't understand regex. Do I need to memorize all the patterns?
**A**: **No!** Start with these basics:
- `\d` = digit (0-9)
- `\w` = word character (letter, digit, underscore)
- `\s` = whitespace (space, tab)
- `+` = one or more
- `?` = zero or one
- `*` = zero or more

**Common patterns you'll use:**
- `r'\d+'` = one or more digits (for phone numbers)
- `r'\w+'` = one or more word characters (for tokenization)
- `r'https?://\S+'` = URLs (http:// or https:// followed by non-space chars)
- `r'\S+@\S+'` = email addresses

**Tip**: Use [regex101.com](https://regex101.com) to test patterns. For now, you can copy common patterns from examples. You'll learn more as you practice.

---

### Q: What if my regex doesn't work?
**A**: Common issues:
1. **Forgot `r` prefix**: Use `r'\d+'` not `'\d+'`
2. **Pattern too strict**: Start simpler, add complexity
3. **Test first**: Use regex101.com before coding
4. **Copy working examples**: Don't write from scratch initially

**Debugging tip**: Test pattern on one example first, then apply to all text.

---

### Q: What are stop words? Why remove them?
**A**: Stop words = common words that don't carry meaning:
- Examples: "the", "a", "an", "is", "are", "and", "or"
- They appear in almost every document
- They add noise, not signal

**Why remove?** Focus on meaningful words. "machine learning" is more important than "the machine learning".

**Exception**: Keep them if word order matters (less common).

---

### Q: What's the difference between stemming and lemmatization?
**A**: 
- **Stemming**: "running" → "run" (just chops off endings)
- **Lemmatization**: "better" → "good" (actually understands meaning)

**For now**: Use stemming if needed. Lemmatization is better but more complex. Many libraries handle this automatically.

---

## Vectors and Text Representation

### Q: What is a vector? I'm confused.
**A**: **Simple answer**: A vector is just a **list of numbers**.

**For text:**
- Each position = a word in the vocabulary
- The number = how many times that word appears (or TF-IDF score)

**Example:**
```
Vocabulary: ["python", "machine", "learning", "data"]
Document: "python python machine"
Vector: [2, 1, 0, 0]
         ↑  ↑  ↑  ↑
        py ma le da (word positions)
```

**Don't think of it as math!** Think: "List of counts for each word."

---

### Q: Why do we need to convert text to numbers?
**A**: Computers can only work with numbers. To use machine learning on text, we need numbers.

**Analogy**: Like translating between languages. We're translating "words" → "numbers" so computers understand.

---

### Q: What's the difference between Bag of Words and TF-IDF?
**A**:
- **Bag of Words**: Counts words. All words have equal weight.
- **TF-IDF**: Weights words by importance. Rare words get higher scores.

**Example:**
- Word "the": Appears in all documents → Low TF-IDF score
- Word "machine learning": Appears rarely → High TF-IDF score

**Key insight**: Rare words are more informative!

---

### Q: The TF-IDF formula looks scary. Do I need to calculate it manually?
**A**: **No!** Use `TfidfVectorizer` from scikit-learn. It does everything for you.

**Understanding > Calculation**:
- TF-IDF = Term Frequency × Inverse Document Frequency
- Intuition: "Rare words are more important"
- You don't need to calculate it yourself!

---

### Q: What's the difference between sparse and dense vectors?
**A**: 

**Sparse Vectors** (like TF-IDF):
- Most values are zero: `[0, 0, 1, 0, 0, 0, 0, 1, 0, ...]`
- Vocabulary size: 10,000-100,000+ dimensions
- Each document uses only a small fraction of vocabulary
- Memory efficient when stored in sparse format
- Interpretable: you know what each dimension means (word frequency)

**Dense Vectors** (like embeddings - Class 3):
- Most/all values are non-zero: `[0.23, -0.15, 0.87, ..., 0.42]`
- Fixed dimension: typically 100-768 dimensions
- Each dimension has learned meaning (not directly interpretable)
- Captures semantic relationships between words

**For TF-IDF**: `TfidfVectorizer` returns a sparse matrix. Use `.toarray()` if you need a regular array. That's it!

---

## Similarity and Search

### Q: What is cosine similarity? I don't understand angles.
**A**: **Simple explanation**: Measures how similar two vectors are.

- Score 1.0 = Identical
- Score 0.8 = Very similar
- Score 0.0 = Unrelated
- Score -1.0 = Opposite

**Don't worry about the angle math!** Just use the function:
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector1, vector2)
```

Higher score = more similar. That's all you need!

---

### Q: What's the difference between keyword and semantic search?
**A**:

| Keyword Search | Semantic Search |
|----------------|-----------------|
| Looks for exact words | Understands meaning |
| "Python" finds only "Python" | "Python" finds "Python", "programming", related terms |
| Fast but limited | Slower but smarter |

**Two types of keyword search:**
1. **Simple matching**: Just checks if words appear (fastest, simplest)
2. **TF-based**: Ranks by Term Frequency - words appearing more often score higher

**Example:**
- Keyword (simple): "ML expert" → finds nothing (no exact match)
- Keyword (TF-based): "machine learning" → ranks by how often these words appear
- Semantic: "ML expert" → finds "machine learning specialist" (understands synonyms)

---

### Q: Which search method should I use?
**A**: 
- **Keyword (simple)**: When you need exact matches and maximum speed
- **Keyword (TF-based)**: When you want ranking by word frequency, still fast
- **Similarity Search (TF-IDF)**: When you want similarity-based ranking (better than keyword but still keyword-based - does NOT handle synonyms)
- **Best**: Start with TF-based keyword for speed, upgrade to semantic if you need better understanding

---

## Clustering

### Q: What is clustering?
**A**: Grouping similar documents together **without labels**.

**Key point**: You don't tell the algorithm what groups to make. It finds them automatically!

**Example**: Grouping CVs into "data scientists", "software engineers", "analysts" - but you didn't label them first.

---

### Q: Is clustering supervised or unsupervised?
**A**: **Unsupervised!** No labels needed. The algorithm discovers patterns on its own.

**Compare**:
- **Supervised** (like Logistic Regression): Need labels ("spam" or "not spam")
- **Unsupervised** (clustering): No labels, just find groups

**Same preprocessing**, different goal!

---

### Q: How many clusters should I use?
**A**: It depends! Common approach:
- Start with 2-5 clusters
- Look at the results
- Adjust based on whether groups make sense

**Rule of thumb**: Not too many (everything separate) or too few (everything together).

---

### Q: Why are my clusters not what I expected?
**A**: Common reasons:
1. **Wrong number of clusters**: Try different K values
2. **Poor preprocessing**: Clean your text better
3. **Need better representation**: TF-IDF might not be enough (try embeddings - Class 3!)

**Tip**: Visualize clusters with PCA to see if they make sense.

---

## Supervised vs Unsupervised

### Q: What's the difference between supervised and unsupervised learning?
**A**: 

| Supervised | Unsupervised |
|------------|--------------|
| Needs labels | No labels needed |
| Predict/classify | Find patterns |
| Example: Spam detection | Example: Clustering |

**Key insight**: Same text preprocessing for both! Same vectors! Only difference: labels.

---

### Q: I learned Logistic Regression in Chapter 0. Is this different?
**A**: **Same algorithm!** Just different input:
- **Chapter 0**: Numbers → Logistic Regression → Class
- **Class 1**: Text → Preprocessing → Numbers → Logistic Regression → Class

**It's the same Logistic Regression, just with text as input!**

---

## General Questions

### Q: I'm getting errors in my code. Help!
**A**: Common issues:

1. **Import errors**: Did you `import pandas as pd`?
2. **File not found**: Check file path. Use `os.path.join('data', 'file.csv')`
3. **Attribute errors**: Check spelling. `df.head()` not `df.Head()`
4. **Type errors**: Make sure you're using the right data type

**Tip**: Read error messages carefully. They usually tell you what's wrong!

---

### Q: The exercises are taking too long. Am I doing something wrong?
**A**: **No!** This is normal, especially if you're new to programming. 

**Tips**:
- It's okay to look at solutions
- Ask for help if stuck > 10 minutes
- Focus on understanding, not perfect code
- Practice makes it faster

**Remember**: Everyone struggles at first. Keep going!

---

### Q: I don't understand how everything connects. Help!
**A**: Here's the big picture:

```
Text (messy) 
  → Preprocessing (clean)
  → Vectors (numbers)
  → Machine Learning (clustering/search)
  → Results (groups/similar documents)
```

**Same pipeline for all NLP tasks!** You're learning the pieces now, they'll connect in Class 3.

---

### Q: What should I focus on most?
**A**: 
1. **Understanding the pipeline**: Text → Preprocessing → Vectors → Task
2. **Vectors concept**: Text becomes numbers (list of counts/scores)
3. **Preprocessing importance**: Clean text = better results
4. **Unsupervised learning**: Clustering finds patterns without labels

Don't worry about perfect regex or memorizing formulas. Understanding the concepts is more important!

---

## Quick Reference

### Most Important Concepts:
1. **Preprocessing matters**: Garbage in = garbage out
2. **Vectors = lists of numbers** representing text
3. **TF-IDF weights rare words higher** (more informative)
4. **Clustering is unsupervised** (no labels needed)
5. **Same preprocessing** for supervised and unsupervised

### When Stuck:
1. Check error messages
2. Test code on small examples first
3. Use library functions (don't write from scratch)
4. Ask for help - it's normal to struggle!

### Key Functions to Remember:
```python
# Preprocessing
re.findall(r'\w+', text)  # Extract words
re.sub(r'https?://\S+', '', text)  # Remove URLs

# Keyword Search (TF-based)
def calculate_tf(text, word):
    words = text.lower().split()
    return words.count(word.lower()) / len(words) if len(words) > 0 else 0

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)  # Returns sparse matrix
vectors_dense = vectors.toarray()  # Convert to dense if needed

# Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(vectors)

# Similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vec1, vec2)
```

---

**Remember**: Understanding concepts > Perfect code. Keep practicing! 🚀

