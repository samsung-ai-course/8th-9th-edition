# Class 3: FAQ - Common Questions & Answers

## Embeddings Basics

### Q: What is an embedding? I'm still confused.
**A**: **Simple answer**: An embedding is a **list of numbers** that represents a word's meaning.

**Key idea**: Words with similar meanings have similar lists of numbers.

**Analogy**: 
- Like a fingerprint for words
- Similar words = similar fingerprints
- The numbers encode meaning (you don't need to understand HOW, just that they do)

**Example:**
```
"king" → [0.2, -0.1, 0.8, 0.3, ...] (300 numbers)
"queen" → [0.3, -0.2, 0.7, 0.4, ...] (similar numbers = similar meaning)
```

---

### Q: Why do we need embeddings if TF-IDF works?
**A**: TF-IDF can't handle synonyms:
- "NLP" and "natural language processing" have similarity ≈ 0 with TF-IDF
- Embeddings understand they mean the same thing (high similarity)

**Use embeddings when**: You need semantic understanding, not just word matching.

**Code example:**
```python
# TF-IDF: Different words = 0 similarity
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
docs = ["NLP is great", "natural language processing is amazing"]
vectors = tfidf.fit_transform(docs)
similarity = cosine_similarity(vectors[0], vectors[1])  # ≈ 0 (no shared words!)

# Embeddings: Understand synonyms
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
similarity = model.similarity('NLP', 'natural_language_processing')  # High similarity!
```

---

### Q: What does "300 dimensions" mean? I can't visualize that.
**A**: **Don't try to visualize 300D!** Think of it like this:

- **2D**: GPS coordinates (lat, lon) - 2 numbers
- **3D**: 3D space (x, y, z) - 3 numbers  
- **300D**: 300 numbers (features) - same concept, more numbers!

**Analogy**: 
- Like a spreadsheet with 300 columns
- Each column = one feature/aspect of meaning
- Similar words = similar values across columns

**You don't need to visualize it!** Just know: each word = 300 numbers.

---

### Q: How do embeddings "know" that "NLP" = "natural language processing"?
**A**: They learned from context during training:
- Both terms appear in similar contexts in training data
- Neural network learned: "similar context = similar meaning"
- Result: Similar embeddings

**For you**: Trust that pre-trained embeddings have learned this. You'll learn HOW in deep learning chapters.

---

### Q: Can I see what an embedding actually looks like?
**A**: Yes! But you'll see 300 numbers, which isn't helpful. Instead:

**Visualize similarities**:
```python
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

# Get embedding (300 numbers)
vec = model['python']
print(f"Shape: {vec.shape}")  # (300,)
print(f"First 5 values: {vec[:5]}")  # [0.123, -0.456, 0.789, ...]

# Find similar words
similar = model.most_similar('python', topn=3)
# [('java', 0.72), ('javascript', 0.68), ('programming', 0.63)]
```

**Use 2D projections** (PCA/t-SNE) to see relationships visually.

---

## Word Embeddings

### Q: What's the difference between Word2Vec, GloVe, and FastText?
**A**: Different training methods, each with strengths:

- **Word2Vec**: Predicts surrounding words (local context)
  - Good for: General purpose, common words
  - Weakness: Poor OOV handling
  
- **GloVe**: Uses word co-occurrence statistics (global patterns)
  - Good for: Large corpus, common words
  - Weakness: Poor OOV handling
  
- **FastText**: Uses subword information (character n-grams)
  - Good for: OOV words, misspellings, rare words
  - Strength: Can handle unknown words!

**Key point**: FastText handles OOV better, but all three work well for common words. Choose based on your needs!

**Code comparison - finding similar words:**
```python
import gensim.downloader as api

# All have similar interface for common operations
word2vec = api.load('word2vec-google-news-300')
glove = api.load('glove-wiki-gigaword-300')
fasttext = api.load('fasttext-wiki-news-subwords-300')

# Finding similar words: All work the same way
word2vec.most_similar('python', topn=3)  # [('java', 0.72), ...]
glove.most_similar('python', topn=3)     # [('java', 0.68), ...]
fasttext.most_similar('python', topn=3)  # [('java', 0.65), ...]
```

---

### Q: What does "king - man + woman ≈ queen" mean?
**A**: This shows embeddings capture relationships:
- "king" and "queen" are related (royalty)
- "man" and "woman" are related (gender)
- The relationship is preserved in embedding space

**It's like**: Royalty relationship + gender swap = new word

**Don't worry if it's confusing!** It's a cool property, but understanding it perfectly isn't required.

**Code example:**
```python
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

# Vector arithmetic
result = model.most_similar(
    positive=['woman', 'king'],
    negative=['man'],
    topn=1
)
# Returns: [('queen', 0.71)] - the relationship is preserved!
```

---

### Q: Why do we use pre-trained embeddings? Can't I train my own?
**A**: **You can**, but:

**Pre-trained advantages**:
- ✅ Trained on billions of words (Wikipedia, web)
- ✅ Ready to use immediately
- ✅ Usually better than training your own
- ✅ No computational cost

**Train your own only if**:
- You have domain-specific text (medical, legal)
- You have massive amounts of data
- You have computational resources

**For now**: Use pre-trained. It's the standard approach.

---

## Sentence Embeddings

### Q: How do you get from word embeddings to sentence embeddings?
**A**: **Two main approaches**:

**1. Simple averaging** (what you learned in Part 1):
```python
import numpy as np
import gensim.downloader as api

model = api.load('word2vec-google-news-300')
sentence = "I love machine learning"
words = sentence.lower().split()

# Get word embeddings
word_vectors = [model[w] for w in words if w in model]

# Average them
sentence_embedding = np.mean(word_vectors, axis=0)
print(f"Sentence embedding shape: {sentence_embedding.shape}")  # (300,)
```

**2. Sentence-BERT** (what you learned in Part 2):
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence = "I love machine learning"
sentence_embedding = model.encode(sentence)
print(f"Sentence embedding shape: {sentence_embedding.shape}")  # (384,)
```

**When to use**: Start with averaging (simple, fast). Use Sentence-BERT for better quality (slower but better).

---

### Q: Does averaging lose information?
**A**: **Yes, a bit**:
- Loses word order ("dog bites man" vs "man bites dog" = same average)
- Treats all words equally
- Doesn't understand sentence structure

**But**: For many tasks (search, clustering), it works well enough!

**Sentence-BERT is better** because:
- ✅ Captures word order
- ✅ Understands sentence structure
- ✅ Trained with contrastive learning (similar sentences pushed together)

**Rule of thumb**: Averaging is fine for exploration. Sentence-BERT for production/classification.

---

## Neural Networks (Brief Intro)

### Q: How are embeddings learned? The neural network thing confuses me.
**A**: **High-level answer** (details in deep learning chapters):

1. Neural network sees millions of sentences
2. Learns patterns: "words in similar contexts = similar meaning"
3. Creates embeddings that capture these patterns

**For now**: 
- ✅ Embeddings are learned using neural networks
- ✅ They learn from data automatically
- ✅ You'll learn HOW in deep learning chapters
- ❌ Don't worry about the details now

**Analogy**: Like a student learning patterns from examples, but much faster.

---

### Q: Do I need to understand neural networks for this class?
**A**: **No!** Just know:
- Embeddings are learned using neural networks
- They learn patterns from data
- You'll learn details in deep learning chapters

**Focus on**: Using embeddings, not creating them.

---

## Installation & Technical

### Q: I can't install spaCy. Help!
**A**: Common solutions:

1. **Use Google Colab**: Pre-installed, no setup needed
2. **Check Python version**: Need Python 3.7+
3. **Use pip**: `pip install spacy`
4. **Download model**: `python -m spacy download en_core_web_md`
5. **Use TF-IDF fallback**: If all else fails, code has fallback

**Don't let installation block you!** Ask for help or use Colab.

---

### Q: My computer is slow with embeddings. What should I do?
**A**: 
- Use smaller models (en_core_web_sm instead of en_core_web_md)
- Process in batches
- Use Google Colab (free GPU access)
- For small datasets, TF-IDF is faster

**It's normal for embeddings to be slower than TF-IDF!**

---

### Q: What if a word isn't in the embedding vocabulary? (OOV)
**A**: This is called "out-of-vocabulary" (OOV):

**Word2Vec/GloVe**: 
- Word not found → Skip it or use zero vector
- Problem: Loses information

**FastText**: 
- ✅ Handles OOV! Uses subwords (character n-grams)
- Example: "unhappiness" = "un" + "happiness" → Gets embedding even if word is unknown
- This is FastText's key advantage!

**For now**: FastText handles OOV best. Word2Vec/GloVe work fine for common words.

**See Troubleshooting section below** for code example on handling OOV words.

---

## Using Embeddings

### Q: When should I use embeddings vs TF-IDF?
**A**: 

| Use Embeddings When | Use TF-IDF When |
|---------------------|-----------------|
| Need semantic understanding | Speed is critical |
| Handling synonyms important | Simple task |
| Related concepts matter | Small dataset |
| Better accuracy needed | Interpretability needed |
| Classification tasks | Exact keyword matching |

**Rule of thumb**: Start with TF-IDF, upgrade to embeddings if you need semantic understanding or better accuracy.

---

### Q: How do I know if embeddings are working better?
**A**: Compare results:
- **Search**: Same query - embeddings find more relevant results (synonyms, related concepts)?
- **Clustering**: Embeddings create better semantic groups?
- **Classification**: Better accuracy with embeddings?
- **Synonyms**: Embeddings match them, TF-IDF doesn't?

**Visual check**: Embeddings should group semantically similar items together.

**Code example - comparing synonyms:**
```python
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

model = api.load('word2vec-google-news-300')

# Test synonym understanding
words = ["great", "excellent", "amazing", "terrible"]
similarities = []
for w1 in words:
    for w2 in words:
        if w1 != w2:
            sim = model.similarity(w1, w2)
            similarities.append((w1, w2, sim))

# Positive words should be similar, negative words different
# "great" ↔ "excellent": ~0.75 (high!)
# "great" ↔ "terrible": ~0.1 (low!)
```

---

### Q: Should I always use embeddings?
**A**: **No!** Consider:
- Speed: TF-IDF is faster
- Interpretability: TF-IDF is easier to understand
- Accuracy: Embeddings are usually better
- Complexity: TF-IDF is simpler
- OOV handling: FastText handles OOV, TF-IDF doesn't need it

**Choose based on your needs!** For classification, embeddings usually win. For simple search, TF-IDF might be enough.

---

## Visualization

### Q: How do you visualize 300-dimensional embeddings?
**A**: **Reduce dimensions**:
- **PCA**: Principal Component Analysis (reduces to 2D/3D)
- **t-SNE**: Better for visualization (but slower)

**What you see**: Similar words cluster together in 2D projection.

**Note**: 2D is just for visualization. Actual embeddings are still 300D.

**Code example:**
```python
import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = api.load('word2vec-google-news-300')
words = ['python', 'java', 'javascript', 'great', 'excellent', 'amazing']
vectors = [model[w] for w in words]

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# Plot
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
plt.show()
# Programming languages cluster together, positive words cluster together!
```

---

### Q: Why don't my visualized embeddings look like they cluster?
**A**: Possible reasons:
- **Small dataset**: Need more data points
- **Poor reduction**: Try different methods (PCA vs t-SNE)
- **Embeddings not great**: Try different model
- **Normal**: Some words just don't cluster well

**Don't worry if visualization isn't perfect!** Focus on search/clustering results.

---

## General Questions

### Q: I'm overwhelmed by all these concepts. Help!
**A**: **Focus on these key points**:
1. Embeddings = lists of numbers representing meaning
2. Similar words = similar numbers
3. Use pre-trained embeddings (don't train your own)
4. Average word embeddings to get sentence embeddings
5. Embeddings understand semantics better than TF-IDF

**Everything else is details you'll learn as you practice!**

---

### Q: Do I need to understand HOW embeddings work?
**A**: **Not yet!** 

**Focus on**:
- ✅ What embeddings are (conceptually)
- ✅ How to use them
- ✅ When they're better than TF-IDF

**Learn later** (deep learning chapters):
- ❌ How neural networks create them
- ❌ Training process
- ❌ Architecture details

**For now**: Use them like a tool. Understand what they do, not how they're made.

---

### Q: Why are embeddings better for clustering?
**A**: They understand meaning:
- "NLP expert" and "natural language processing specialist" cluster together
- "data scientist" and "ML engineer" cluster together
- TF-IDF would treat these as different (different words)

**Result**: Better semantic grouping!

---

### Q: Can embeddings handle context? Like "bank" (river vs money)?
**A**: **Basic embeddings (Word2Vec, GloVe)**: Limited
- "bank" has one embedding regardless of context
- Can confuse river bank vs money bank

**Transformers (BERT, next class)**: Better
- Context-aware embeddings
- "river bank" vs "money bank" = different embeddings

**For now**: Basic embeddings work well for most tasks!

---

## Preprocessing & Classification

### Q: How does preprocessing affect embeddings differently than TF-IDF?
**A**: **Key difference**:

**TF-IDF**: More preprocessing usually helps
- Removes noise, normalizes text
- Stopword removal, lemmatization improve results

**Embeddings**: Sometimes LESS is more!
- Embeddings learn from context
- Over-cleaning removes important context
- Example: "not good" needs both words - removing "not" loses negation!

**Rule of thumb**: 
- With TF-IDF: Clean aggressively
- With embeddings: Clean lightly (lowercase, basic punctuation removal)

---

### Q: Why does preprocessing sometimes hurt embeddings?
**A**: **Context matters!**

- **Stopword removal**: Can remove important words like "not", "very"
- **Lemmatization**: Can lose subtle differences ("better" vs "best")
- **Over-cleaning**: Removes context that embeddings learned from

**Example**: 
- "This movie is not good" → Removing "not" makes it positive!
- Embeddings need context to understand negation

**Solution**: Test different preprocessing strategies. Sometimes baseline (minimal preprocessing) works best!

---

### Q: How do I use embeddings for classification?
**A**: **Simple pipeline**:

1. **Load pre-trained embeddings** (Word2Vec, GloVe, FastText)
2. **Create document embeddings** (average word embeddings)
3. **Train classifier** (Logistic Regression from Chapter 0)
4. **Evaluate** (accuracy, precision, recall, F1)

**Example**:
```python
# 1. Load embeddings
import gensim.downloader as api
model = api.load('word2vec-google-news-300')

# 2. Create document embeddings (average words)
def doc_to_vector(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

# 3. Train classifier
from sklearn.linear_model import LogisticRegression
X = [doc_to_vector(text, model) for text in texts]
classifier = LogisticRegression()
classifier.fit(X, y)

# 4. Evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

**Key insight**: Embeddings transform text → dense vectors → classification works!

---

### Q: What's the difference between word averaging and Sentence-BERT for classification?
**A**: 

**Word Averaging**:
- ✅ Simple and fast
- ✅ Works reasonably well
- ❌ Loses word order
- ❌ Treats all words equally

**Sentence-BERT**:
- ✅ Captures word order and context
- ✅ Usually better accuracy
- ✅ Trained specifically for sentences
- ❌ Slower than averaging

**When to use**:
- **Averaging**: Quick experiments, large datasets, speed matters
- **Sentence-BERT**: Better accuracy needed, production systems

**In Part 2**: You'll see Sentence-BERT usually outperforms averaging!

**Code comparison - embedding shapes:**
```python
import numpy as np
import gensim.downloader as api
from sentence_transformers import SentenceTransformer

word_model = api.load('word2vec-google-news-300')
sbert = SentenceTransformer('all-MiniLM-L6-v2')

text = "I love this movie"

# Word averaging: 300 dimensions (from Word2Vec)
words = text.lower().split()
word_vecs = [word_model[w] for w in words if w in word_model]
avg_vec = np.mean(word_vecs, axis=0)  # Shape: (300,)

# Sentence-BERT: 384 dimensions (optimized for sentences)
sbert_vec = sbert.encode(text)  # Shape: (384,)

# Sentence-BERT usually gives better classification results!
```

---

## Linguistic Features (POS & NER)

### Q: What is POS tagging and why does it matter?
**A**: **POS = Part-of-Speech** (noun, verb, adjective, adverb, etc.)

**Why it matters for sentiment**:
- Adjectives and adverbs often carry sentiment
- Example: "absolutely terrible" (ADV + ADJ) → both words matter
- Filtering to ADJ+ADV can focus on sentiment-bearing words

**Does it help?**:
- Sometimes yes (reduces noise)
- Sometimes no (loses context like "not good")
- **Test it!** Results vary by dataset

**Key insight**: Context matters - "not good" needs both words!

---

### Q: What is NER and when does it help vs hurt?
**A**: **NER = Named Entity Recognition** (identifies PERSON, ORGANIZATION, LOCATION, etc.)

**When entities help**:
- Entity IS the sentiment: "I love Apple" (entity is object of sentiment)
- Provides context: "Tom Hanks was amazing" (actor name adds context)

**When entities hurt**:
- Entity distracts: "Apple products are expensive" (entity is neutral, sentiment is about products)
- Too many entities: Overwhelms sentiment words

**Does removing entities help?**:
- **Test it!** Results vary
- Sometimes removing entities focuses on sentiment words
- Sometimes keeping entities provides important context

**Rule of thumb**: Test both ways. No universal answer!

---

## Semantic Search

### Q: How is semantic search different from keyword search?
**A**: 

**Keyword search (TF-IDF)**:
- Finds exact word matches
- "Python developer" → finds only documents with "Python" and "developer"
- Misses: "Python programmer", "Python coder"

**Semantic search (embeddings)**:
- Finds meaning matches
- "Python developer" → finds "Python programmer", "Python coder", "Python engineer"
- Understands synonyms and related concepts!

**Example**:
- Query: "amazing action movie"
- Keyword search: Only finds exact phrase
- Semantic search: Finds "fantastic action film", "incredible action flick", "great action cinema"

**Key insight**: Semantic search understands meaning, not just keywords!

---

### Q: How do I implement semantic search with embeddings?
**A**: **Simple approach**:

1. **Create embeddings** for all documents
2. **Create embedding** for query
3. **Find most similar** documents (cosine similarity)
4. **Return top-k** results

```python
from sklearn.metrics.pairwise import cosine_similarity

# 1. Document embeddings (already created)
doc_embeddings = [doc_to_vector(doc, model) for doc in documents]

# 2. Query embedding
query_embedding = doc_to_vector("amazing action movie", model)

# 3. Find similarities
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# 4. Top-k results
top_k = 5
top_indices = np.argsort(similarities)[::-1][:top_k]

# Return top documents
results = [documents[i] for i in top_indices]
```

**Key insight**: Same similarity search as TF-IDF, but with semantic embeddings!

---

## Quick Reference

### Most Important Concepts:
1. **Embeddings = numbers representing meaning**
2. **Similar words = similar embeddings**
3. **Use pre-trained** (don't train your own)
4. **Word2Vec/GloVe/FastText**: Different methods, FastText handles OOV best
5. **Average word embeddings** for sentences (or use Sentence-BERT)
6. **Embeddings > TF-IDF** for semantic understanding and classification
7. **Preprocessing**: Less is more with embeddings (context matters!)

### When Stuck:
1. Use pre-trained models (gensim for Word2Vec/GloVe/FastText, sentence-transformers for Sentence-BERT)
2. Start with simple averaging for sentence embeddings
3. Compare with TF-IDF to see improvement
4. Test different preprocessing strategies (sometimes baseline works best!)
5. Use Google Colab if installation fails

### Key Functions:
```python
# Load Word2Vec/GloVe/FastText
import gensim.downloader as api
word2vec = api.load('word2vec-google-news-300')
glove = api.load('glove-wiki-gigaword-300')
fasttext = api.load('fasttext-wiki-news-subwords-300')

# Get word embedding
word_vec = word2vec['python']

# Get similar words
similar = word2vec.most_similar('python', topn=5)

# Create document embedding (average words)
def doc_to_vector(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

# Load Sentence-BERT
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer('all-MiniLM-L6-v2')
sentence_vec = sbert.encode("I love machine learning")

# Semantic search
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_vec], doc_vectors)[0]
top_k_indices = np.argsort(similarities)[::-1][:5]
```

---

## Troubleshooting

### Q: Models are downloading slowly or failing. What should I do?
**A**: 

**First-time downloads are large**:
- Word2Vec: ~1.6GB
- GloVe: ~400MB
- FastText: ~900MB
- Sentence-BERT: ~90MB

**Solutions**:
1. **Be patient**: First download takes time
2. **Check internet**: Stable connection needed
3. **Use Colab**: Pre-downloaded or faster downloads
4. **Download once**: Models are cached locally after first download

**If download fails**:
- Try again (network issues)
- Use smaller models if available
- Use Colab (pre-installed models)

---

### Q: I'm getting "KeyError" or "word not in vocabulary" errors.
**A**: 

**For Word2Vec/GloVe**:
- Word not in vocabulary → Skip it or use zero vector
- Common with rare words, proper nouns, misspellings

**Solution**: Use FastText! It handles OOV words better.

**Example**:
```python
# Word2Vec - fails for OOV
try:
    vec = word2vec['misunderstood']  # Might fail
except KeyError:
    vec = np.zeros(300)  # Fallback

# FastText - handles OOV!
vec = fasttext['misunderstood']  # Works! Uses subwords
```

---

### Q: Classification accuracy is low. What should I do?
**A**: **Check these**:

1. **Preprocessing**: Try different strategies (sometimes less is more!)
2. **Embedding method**: Try Word2Vec, GloVe, FastText, Sentence-BERT
3. **Dataset size**: More data usually helps
4. **OOV handling**: Use FastText if many unknown words
5. **Sentence embeddings**: Try Sentence-BERT instead of word averaging

**Common issues**:
- Over-preprocessing (removed important context)
- Too many OOV words (use FastText)
- Poor word averaging (try Sentence-BERT)

**Debug**: Compare preprocessing strategies, try different embedding methods!

---

### Q: Semantic search isn't finding relevant results.
**A**: **Check these**:

1. **Query preprocessing**: Match document preprocessing
2. **Embedding method**: Try different methods (Word2Vec vs GloVe vs FastText)
3. **Similarity threshold**: Results might be too different
4. **Dataset quality**: Embeddings reflect training data quality

**Common issues**:
- Query and documents preprocessed differently
- Using wrong embedding method for domain
- Results are actually relevant but look different (semantic vs keyword)

**Debug**: Check similarity scores, try different queries, compare with TF-IDF baseline!

---

**Remember**: Understanding conceptually > Understanding technically. You'll learn the technical details in deep learning! 🚀

**Key Takeaways**:
- Embeddings understand meaning (semantics)
- FastText handles OOV best
- Preprocessing: Less is more with embeddings
- Sentence-BERT > word averaging for classification
- Test different strategies - no universal "best"!

