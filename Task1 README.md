# YouTube Comment Sentiment Classifier (Marvel's Wolverine Launch)

This project is a narrow AI system: a machine learning classifier built for
one task, sentiment classification of YouTube comments.

## Problem
Classify YouTube comments on PlayStation's video of Marvel's Wolverine
launch at the Sphere in Las Vegas as **Positive**, **Negative**, or **Neutral**.

Video: "Wolverine slashes his way onto @spherevegas to celebrate the launch
of Marvel's Wolverine" (https://www.youtube.com/watch?v=ZvbifV4mKb0)

## User
Community managers and game marketers who want a quick read on audience
reaction without reading every comment by hand.

## Data Source
- 100 public YouTube comments collected with the `youtube-comment-downloader`
  library from the video above.
- I labeled them by hand and removed 16 spam or unreadable comments,
  leaving **83 labeled comments**.
- Only comment text was kept (no usernames or personal data).

**Labeling rules**
- Positive: excitement or praise
- Negative: complaints, disappointment, or sarcasm
- Neutral: questions or facts

**Label counts:** Negative 57, Neutral 20, Positive 6

## Constraints
- Free tools only (Google Colab, scikit-learn)
- No personal data stored
- Very small dataset (83 labeled comments)
- Prediction takes under 1 second

## Model
TF-IDF features (unigrams and bigrams) + Logistic Regression with balanced
class weights. 80/20 stratified train/test split (66 train, 17 test).

## Evaluation
| Metric | Result |
|---|---|
| Accuracy | 58.8% |
| Macro F1 | 0.32 |
| F1 (Negative) | 0.72 |
| F1 (Neutral) | 0.25 |
| F1 (Positive) | 0.00 |

The test set has only 17 comments (12 Negative, 4 Neutral, 1 Positive), so
these scores are preliminary. For further evaluation, 5-fold
cross-validation can be used.

## Success Criteria
- Target: at least 70% accuracy and 0.60 macro F1 on a held-out test set
- Current result: 58.8% accuracy, 0.32 macro F1 (below target)
- Next step: collect more comments, especially Positive ones, to reach
  the target

## Limitations
- The dataset is small and heavily imbalanced (mostly Negative), so the
  model mostly predicts Negative.
- The test set had only 1 Positive comment, so the Positive score is not
  meaningful.
- Many comments are off-topic complaints about physical game media rather
  than reactions to the video, which makes the data skew Negative.
- Sarcasm is hard to detect, and all comments come from one video.
- Results are preliminary.

## Future Work
- Collect more comments, especially Positive ones, to balance the dataset
- Use 5-fold cross-validation for a more reliable evaluation on small data
- Test other models (e.g., a pretrained transformer) and compare F1 scores

## Files
- `comments_labeled.csv`: labeled dataset
- `notebook.ipynb`: data collection, labeling, and model code
- `README.md`: this document
