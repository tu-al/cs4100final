# cs4100final
# Detecting Automated and Deceptive Online Reviews

This repo implements a modular, transparent pipeline to distinguish genuine and bot-generated online reviews using interpretable textual and behavioral signals.

## Goals

- Construct interpretable feature families:
  - Textual: n-grams, punctuation patterns, length, lexical diversity, etc.
  - Behavioral: account age, rating variance, cadence of posting, etc.
- Train simple, inspectable decision functions (linear models, margin-based rules, anomaly scores) implemented from first principles.
- Evaluate domain transfer (e.g., electronics → hospitality) and robustness to “humanizing” strategies.
- Provide human-understandable explanations for why a review was flagged.

## Repo layout

See the directory overview in this README for high-level structure. Core code is under `src/review_auth/`.

## Getting started

```bash
# create environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# install project in editable mode
pip install -e .
