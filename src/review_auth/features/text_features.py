# n-grams, punctuation, length, entropy
"""
text_features.py — convert review text into numeric information

Purpose:
turn each review’s text into simple numbers that describe how its written.
These features help the model tell apart natural, honest reviews from ones that
look copy-pasted or fake

Feature types:
1. n-grams
   - small word or character sequences ( "great product", "not good")
   - show patterns in phrasing.

2. punctuation
   - count of !, ?, and other marks
   - fakes often overuse exclamation points or odd punctuation

3. length
   - total number of words or characters.
   - very short or extremely long reviews may look suspicious

4. entropy
   - measures how repetitive or diverse the words are
   - low entropy = repetitive (often fake), high entropy = more natural

Main functions to implement:
- clean_text(text): lowercase, remove extra symbols, split into words (if not done in preprocessing)
- extract_ngrams(text, n=1 or 2): return dictionary/counts of n-grams
- punctuation_features(text): count punctuation marks and their ratios.
- length_features(text): count words and characters
- entropy_feature(text): compute word diversity using simple probability
- build_text_features(reviews): run all steps above and return one combined list
  of features per review
"""
