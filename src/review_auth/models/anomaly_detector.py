# simple anomaly scoring schemes
"""
anomaly_detector.py — detecting unusual or suspicious reviews (unsupervised)

Purpose:
This file adds a second kind of signal: reviews that just look strange,
even if the model didn’t catch them directly. It doesn’t learn from labels
it checks for weird patterns that might mean fake behavior.

How it works
1. Find near duplicate reviews:
   - Two reviews that are almost the same (same wording, structure)
   - These might be auto-generated or copied by bots
2. Detect text outliers:
   - very short reviews
   - Too many punctuation marks or emojis
   - Repeated phrases or patterns

main functions:
- near_duplicate_score(texts): compare each review with others and return
  a score (higher = more similar to existing reviews).
- outlier_score(lengths, punctuation_rate): flag reviews that look extreme
  compared to the rest.

"""
