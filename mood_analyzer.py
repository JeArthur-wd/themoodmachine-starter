# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the meaning of a sentiment word that follows them.
NEGATORS = {
    "not", "no", "never", "dont", "don't", "isnt", "isn't", "cant", "can't",
    "wont", "won't", "didnt", "didn't", "wasnt", "wasn't", "aint", "ain't",
}
# How many tokens after a negator can still be flipped ("not very happy").
NEGATION_WINDOW = 2

# Words, ASCII emoticons, or any single non-ASCII character (an emoji).
TOKEN_PATTERN = re.compile(r":-?\)|:-?\(|[a-z0-9']+|[^\x00-\x7f]")


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Strip whitespace, lowercase, and straighten curly apostrophes
          - Pull out words, ASCII emoticons (":)", ":(") and emojis as tokens;
            other punctuation is dropped
          - Normalize stretched words ("soooo" -> "soo")
        """
        cleaned = text.strip().lower().replace("’", "'")
        tokens = TOKEN_PATTERN.findall(cleaned)
        tokens = [re.sub(r"(.)\1{2,}", r"\1\1", t) for t in tokens]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _find_hits(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Return (positive_hits, negative_hits) for a list of tokens.

        Negation: a negator flips the next sentiment word within
        NEGATION_WINDOW tokens, so "not happy" counts as negative and
        "not bad" counts as positive.
        """
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negation_left = 0

        for i, token in enumerate(tokens):
            # "no cap" is slang for "for real", not a negation.
            is_no_cap = token == "no" and tokens[i + 1:i + 2] == ["cap"]
            if token in NEGATORS and not is_no_cap:
                negation_left = NEGATION_WINDOW
                continue

            is_positive = token in self.positive_words
            is_negative = token in self.negative_words

            if is_positive or is_negative:
                if negation_left > 0:
                    is_positive, is_negative = is_negative, is_positive
                    token = f"not {token}"
                    negation_left = 0
                if is_positive:
                    positive_hits.append(token)
                else:
                    negative_hits.append(token)
            elif negation_left > 0:
                negation_left -= 1

        return positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.
        Negated words count the opposite way (see _find_hits).
        """
        positive_hits, negative_hits = self._find_hits(self.preprocess(text))
        return len(positive_hits) - len(negative_hits)

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The mapping is:
          - hits on both sides -> "mixed" (even if the score cancels to 0)
          - score > 0          -> "positive"
          - score < 0          -> "negative"
          - otherwise          -> "neutral"
        """
        positive_hits, negative_hits = self._find_hits(self.preprocess(text))
        score = len(positive_hits) - len(negative_hits)

        if positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Example: 'Score = 2 (positive: ['love', 'great'], negative: [])'
        """
        positive_hits, negative_hits = self._find_hits(self.preprocess(text))
        score = len(positive_hits) - len(negative_hits)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
