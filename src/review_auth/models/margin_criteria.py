# margin-based decision rules
"""
margin_criteria.py — turning probabilities into clear decisions

Purpose:
After training the model, it will give each review a probability that it’s fake
This file decides what to do with that probability. for example, what value of
probability counts as “fake enough” to flag.

How it works:
- The model gives you a number like 0.92 or 0.23 for each review.
- We choose a threshold (like 0.8)
- If probability ≥ 0.8, call it FAKE.
- Otherwise, call it REAL.

main functions to implement
-----------------------------
- decide(prob, threshold): return 1 for fake, 0 for real.
"""
