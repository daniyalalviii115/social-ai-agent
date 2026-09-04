"""
ml/naive_bayes.py — Multinomial Naive Bayes Sentiment Analyzer
===============================================================
Phase 3 of the Autonomous Social Media AI Agent project.

Trains a TF-IDF + Multinomial Naive Bayes sentiment classifier on the
synthetic comment data produced by Phase 2 (core/simulator.py).

Downstream consumers:
  - Phase 5 (ml/fuzzy_engine.py)        : avg_sentiment_score per post feeds
                                           the fuzzy inference system as one
                                           of its crisp input variables.
  - Phase 6 (ml/genetic_algorithm.py)   : fitness function uses predicted
                                           sentiment alongside engagement_rate.
  - Phase 7 (dashboard/app.py)          : real-time single-comment inference
                                           for the Strategy Advisor UI panel.

Usage:
    python ml/naive_bayes.py
    from ml.naive_bayes import SentimentAnalyzer, train_and_save_model
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402


class SentimentAnalyzer:
    """
    Multinomial Naive Bayes sentiment classifier with TF-IDF feature extraction.

    Labels:
        "positive"  ->  comment expresses approval / enthusiasm
        "neutral"   ->  informational or ambivalent comment
        "negative"  ->  comment expresses disapproval / criticism

    The classifier is trained on the synthetic ``comments.csv`` produced by
    Phase 2.  Because the data was generated with realistic sentiment
    distributions correlated to engagement_rate, the trained model captures
    meaningful lexical patterns (not just noise), enabling it to generalise
    to unseen comment text fed through the Phase 7 dashboard.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Converts raw comment strings to TF-IDF feature matrices.
    model : MultinomialNB
        The fitted Naive Bayes classifier.
    is_fitted : bool
        Set to True after a successful call to :meth:`train`.
    classes_ : list[str]
        Ordered list of class labels, set after training.
    """

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            lowercase=True,
            strip_accents="unicode",
        )
        self.model: MultinomialNB = MultinomialNB(alpha=0.5)
        self.is_fitted: bool = False
        self.classes_: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, comments_df: pd.DataFrame) -> dict:
        """
        Fit the vectorizer and classifier on labelled comment data.

        Parameters
        ----------
        comments_df : pd.DataFrame
            Must contain columns ``comment_text`` and ``true_sentiment_label``.

        Returns
        -------
        dict
            Keys:
            - ``accuracy``               : float  — test-set accuracy
            - ``classification_report``  : dict   — per-class precision/recall/F1
            - ``confusion_matrix``       : list[list[int]] — rows=true, cols=pred
            - ``test_size``              : int    — number of test samples

        Raises
        ------
        ValueError
            If required columns are missing or fewer than 3 classes are present.
        """
        required = {"comment_text", "true_sentiment_label"}
        missing  = required - set(comments_df.columns)
        if missing:
            raise ValueError(
                f"comments_df is missing required columns: {missing}"
            )

        n_classes = comments_df["true_sentiment_label"].nunique()
        if n_classes < 2:
            raise ValueError(
                f"Need at least 2 sentiment classes to train; found {n_classes}."
            )

        X = comments_df["comment_text"].astype(str).values
        y = comments_df["true_sentiment_label"].astype(str).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            stratify=y,
            random_state=config.RANDOM_SEED,
        )

        # Fit
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf  = self.vectorizer.transform(X_test)
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred   = self.model.predict(X_test_tfidf)
        acc      = float(accuracy_score(y_test, y_pred))
        report   = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        cm       = confusion_matrix(
            y_test, y_pred, labels=self.model.classes_
        ).tolist()

        self.is_fitted  = True
        self.classes_   = list(self.model.classes_)

        return {
            "accuracy":               acc,
            "classification_report":  report,
            "confusion_matrix":       cm,
            "test_size":              int(len(y_test)),
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been trained yet."""
        if not self.is_fitted:
            raise RuntimeError(
                "SentimentAnalyzer has not been trained. "
                "Call .train(comments_df) first."
            )

    def predict_sentiment(self, text: str) -> dict:
        """
        Predict the sentiment label and confidence for a single comment string.

        Parameters
        ----------
        text : str
            Raw comment text (any length).

        Returns
        -------
        dict
            Keys:
            - ``text``                : str   — original input
            - ``predicted_label``     : str   — "positive" | "neutral" | "negative"
            - ``confidence``          : float — probability of the predicted class
            - ``class_probabilities`` : dict  — {class: probability} for all classes
        """
        self._check_fitted()
        vec   = self.vectorizer.transform([str(text)])
        proba = self.model.predict_proba(vec)[0]
        idx   = int(np.argmax(proba))
        return {
            "text":               text,
            "predicted_label":    self.classes_[idx],
            "confidence":         float(proba[idx]),
            "class_probabilities": {
                cls: float(p) for cls, p in zip(self.classes_, proba)
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiments for a list of comment strings.

        Parameters
        ----------
        texts : list[str]
            Raw comment texts to classify.

        Returns
        -------
        list[dict]
            One ``predict_sentiment`` result dict per input text, in the same
            order as ``texts``.
        """
        self._check_fitted()
        return [self.predict_sentiment(t) for t in texts]

    def score_post_sentiment(
        self, post_id: int, comments_df: pd.DataFrame
    ) -> dict:
        """
        Compute an aggregate sentiment score for a single post.

        The ``avg_sentiment_score`` is computed by mapping predicted labels
        to numeric values: positive -> +1.0, neutral -> 0.0, negative -> -1.0,
        then taking the mean over all comments for that post.  This scalar is
        used as a crisp input to the Phase 5 Fuzzy Inference System.

        Parameters
        ----------
        post_id      : int           — the post whose comments to score.
        comments_df  : pd.DataFrame  — must contain ``post_id`` and
                                       ``comment_text`` columns.

        Returns
        -------
        dict
            Keys:
            - ``post_id``             : int
            - ``avg_sentiment_score`` : float — range [-1.0, +1.0]
            - ``positive_ratio``      : float — fraction of positive predictions
            - ``negative_ratio``      : float — fraction of negative predictions
            - ``neutral_ratio``       : float — fraction of neutral predictions
            - ``comment_count``       : int
        """
        self._check_fitted()

        _default = {
            "post_id":             post_id,
            "avg_sentiment_score": 0.0,
            "positive_ratio":      0.0,
            "negative_ratio":      0.0,
            "neutral_ratio":       0.0,
            "comment_count":       0,
        }

        subset = comments_df[comments_df["post_id"] == post_id]
        if subset.empty:
            return _default

        texts      = subset["comment_text"].astype(str).tolist()
        predictions = self.predict_batch(texts)
        labels      = [p["predicted_label"] for p in predictions]

        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        scores    = [label_map.get(lbl, 0.0) for lbl in labels]

        n            = len(labels)
        pos_ratio    = labels.count("positive") / n
        neg_ratio    = labels.count("negative") / n
        neu_ratio    = labels.count("neutral")  / n

        return {
            "post_id":             post_id,
            "avg_sentiment_score": float(np.mean(scores)),
            "positive_ratio":      round(pos_ratio, 6),
            "negative_ratio":      round(neg_ratio, 6),
            "neutral_ratio":       round(neu_ratio, 6),
            "comment_count":       n,
        }


# ===========================================================================
# Module-level convenience function
# ===========================================================================

def train_and_save_model(
    comments_df: Optional[pd.DataFrame] = None,
) -> "SentimentAnalyzer":
    """
    Load data (if not provided), train, print report, and return the fitted
    ``SentimentAnalyzer`` instance.

    Parameters
    ----------
    comments_df : pd.DataFrame, optional
        If None, loads ``comments.csv`` from ``config.DATA_DIR``.

    Returns
    -------
    SentimentAnalyzer
        Fitted instance ready for inference.
    """
    if comments_df is None:
        path = os.path.join(config.DATA_DIR, "comments.csv")
        print(f"Loading comments from: {path}")
        comments_df = pd.read_csv(path)

    analyzer = SentimentAnalyzer()
    results  = analyzer.train(comments_df)

    # Pretty-print training summary
    print("=" * 60)
    print("  NAIVE BAYES SENTIMENT ANALYZER — TRAINING REPORT")
    print("=" * 60)
    print(f"  Total comments   : {len(comments_df)}")
    print(f"  Test set size    : {results['test_size']}")
    print(f"  Test accuracy    : {results['accuracy']:.4f}  "
          f"({results['accuracy'] * 100:.2f}%)")
    print()
    print("  Per-class metrics (test set):")
    cr = results["classification_report"]
    for cls in analyzer.classes_:
        m = cr.get(cls, {})
        print(f"    {cls:<10}  "
              f"precision={m.get('precision', 0):.3f}  "
              f"recall={m.get('recall', 0):.3f}  "
              f"f1={m.get('f1-score', 0):.3f}  "
              f"support={int(m.get('support', 0))}")
    print()
    print("  Confusion matrix (rows=true, cols=predicted):")
    print(f"  Classes: {analyzer.classes_}")
    for row in results["confusion_matrix"]:
        print(f"    {row}")
    print("=" * 60)

    return analyzer


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    comments_path = os.path.join(config.DATA_DIR, "comments.csv")
    df = pd.read_csv(comments_path)

    analyzer = train_and_save_model(df)

    # Demonstrate score_post_sentiment on post_id=1
    print()
    print("  Demo: score_post_sentiment(post_id=1)")
    score = analyzer.score_post_sentiment(post_id=1, comments_df=df)
    for k, v in score.items():
        print(f"    {k}: {v}")
    print()
