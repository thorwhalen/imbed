# imbed.components.vectorization

Vectorization functions for converting text to embeddings

### Functions

| [`constant_vectorizer`](#imbed.components.vectorization.constant_vectorizer)(segments, \*[, sleep_s])   | Generate basic constant vector for each segment                                         |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`simple_text_embedder`](#imbed.components.vectorization.simple_text_embedder)(texts[, stopwords])       | Extracts a set of lightweight, linguistically significant features from a text segment. |
| [`three_text_features`](#imbed.components.vectorization.three_text_features)(text)                      | Calculate simple (pseudo-)semantic features of the text.                                |

### imbed.components.vectorization.constant_vectorizer(segments, , sleep_s=0)

Generate basic constant vector for each segment

### imbed.components.vectorization.simple_text_embedder(texts, stopwords=None)

Extracts a set of lightweight, linguistically significant features from a text segment.

The function computes several features based on the input text including:

> - Total number of words.
> - Mean and median word lengths.
> - Number and ratio of stopwords.
> - Number of punctuation characters.
> - Total number of characters.
> - Number of sentences and average words per sentence.
> - Lexical diversity (ratio of unique words to total words).
> - Number of numeric tokens.
> - Number of capitalized words.
* **Parameters:**
  * **text** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text segment to analyze.
  * **stopwords** ([*set*](https://docs.python.org/3/builtins/stdtypes.html#set) *,* *optional*) – A set of stopwords for counting. Defaults to an empty set.
* **Returns:**
  A list of numerical features representing the text, in the following order:
  : [num_words, mean_word_length, median_word_length, num_stopwords, stopword_ratio,
    num_punctuation, num_characters, num_sentences, avg_words_per_sentence,
    lexical_diversity, num_numeric_tokens, num_capitalized_words]
* **Return type:**
  [*list*](https://docs.python.org/3/builtins/stdtypes.html#list)

### Example

```pycon
>>> sample_text = "Hello, world! This is an example text, with 123 numbers and various punctuation marks."
>>> sample_stopwords = {'this', 'is', 'an', 'with', 'and'}
>>> simple_text_embedder(sample_text, sample_stopwords)
[14, 5.21..., 6, 5, 0.35..., 4, 86, 2, 7.0, 1.0, 1, 2]
```

### imbed.components.vectorization.three_text_features(text)

Calculate simple (pseudo-)semantic features of the text.
This is meant to be used as a placeholder vectorizer (a.k.a. embedding function) for
text segments, for testing mainly.

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

```pycon
>>> three_text_features("Hello, world!")
(2, 12, 2)
```
