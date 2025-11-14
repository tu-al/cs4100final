"""
linear_classifier.py — simple logistic regression built from scratch

Purpose:
This file will hold the main model that learns to tell if a review is real or fake.
We will build a simple logistic regression model ourselves, without using libraries
like scikit-learn.

The idea:
- Each review is turned into a list of numbers (features), like how many words,
  punctuation marks, or special tokens it has.
- The model learns a set of weights for these features that help separate fake
  and real reviews.
- The output is a number between 0 and 1: the probability that the review is fake.

How it works:
1. Compute a score for each review using:
      score = w1*x1 + w2*x2 + ... + wn*xn + b
   where w = weights we learn, x = features
2. Pass this score through the sigmoid function
3. Compare this prediction to the true label (0 or 1).
4. Adjust the weights (using gradient descent) to reduce error

Main functions to implement
- initialize_weights(n_features): start weights and bias at small random values
- sigmoid(z): compute 1 / (1 + exp(-z))
- predict_prob(X): return probabilities using current weights
- predict(X, threshold=0.5): return 1 if probability ≥ threshold, else 0
- train(X, y, learning_rate, epochs): use gradient descent to update weights

Expected output
- Trained weights (w) and bias (b)
- a function to predict the probability that a review is fake
- a way to test the accuracy of the model
"""