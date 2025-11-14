# combine feature families, feature configs
"""
feature_settings.py — Feature configuration and combination
===========================================================

Purpose:
central place to manage which feature families (text, meta) are used
and how their outputs are combined into one

What it does:
- stores a simple config dict (which features are on/off, key parameters)
- keeps feature order consistent between training and testing
- combines feature blocks (text + meta) into one final array.

Example:
FEATURE_CONFIG = {
    "use_text": True,
    "use_meta": True,
    "text": {"ngrams": (1,2), "use_punct": True},
    "meta": {"use_account_age": True, "use_cadence": True}
}
"""
