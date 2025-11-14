"""
meta_features.py — Reviewer behavior and context features
Purpose:
this file builds "meta" features: numbers that describe how a reviewer behaves
rather than what words they write. These patterns help the system recognize
accounts that act unnaturally, even if their text looks normal

Features in this file:
1️ Account Age
   - howw long the user account existed before posting the review
   - new accounts that instantly post are suspicious
   - computed as: days_since_account_creation = review_date - account_created_date (if we can access that)

2️ rating variance
   - How consistent the reviewer’s star ratings are across all their reviews
   - Always giving 5 stars (zero variance) can indicate fake or paid accounts
   - computed as: variance of all ratings left by this user

3️ cadence
   - How often a reviewer posts reviews
   - posting many reviews within short time gaps looks automated
   - computed as: average time gap between reviews by the same user


main functions to implement:
- account_age_feature(reviews): compute number of days between account creation
  and review posting
- rating_variance_feature(user_reviews): compute variance of ratings per user
- cadence_feature(user_reviews): compute average posting gap per user
- build_meta_features(reviews): combine all of the above into a single
  list or array of numeric features per review
"""
