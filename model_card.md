# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
I used and compared both models on the same 17 labeled posts.

**Intended purpose:**
Classify short, social-media-style text snippets as `positive`, `negative`, `neutral`, or `mixed`. This is a classroom exercise, not a tool for judging anyone's real wellbeing.

**How it works (brief):**
- *Rule based:* text is lowercased and split into tokens (words, emoticons, emojis). Each token found in `POSITIVE_WORDS` adds a positive hit and each token in `NEGATIVE_WORDS` adds a negative hit. A negator ("not", "never", "don't", ...) flips the next sentiment word within two tokens. The label comes from the hits: both kinds present -> `mixed`; otherwise score > 0 -> `positive`, score < 0 -> `negative`, else `neutral`.
- *ML:* bag-of-words counts (`CountVectorizer`) fed to a logistic regression that learns word-to-label weights from `SAMPLE_POSTS` / `TRUE_LABELS`.

## 2. Data

**Dataset description:**
17 posts: the 6 starter posts plus 11 I added. Label counts: 6 positive, 5 negative, 3 mixed, 3 neutral. `SAMPLE_POSTS` and `TRUE_LABELS` are parallel lists and must stay the same length (`zip` would silently drop extras in `main.py`, and `ml_experiments.py` raises a `ValueError` on a mismatch).

**Labeling process:**
I labeled each post by how I read the writer's overall feeling, not by which words appear. Sarcastic posts are labeled by their intended meaning ("I absolutely love getting stuck in traffic" -> negative). Hard or arguable labels:
- `"I'm fine 🙂"` -> I chose `neutral`, but it could easily be a masked negative.
- `"that concert was sick, no cap 🔥"` and `"this food is fire 🔥"` -> `positive`, because that is the slang meaning.
- `"just got the job!!! sooo happy 😭"` -> `positive`; the crying emoji is happy tears.
- `"nothing special, just another day"` -> `neutral`, but someone could read it as mildly negative.

**Important characteristics of your dataset:**
Slang ("lowkey", "no cap", "sick", "fire"), emojis (🔥 😩 😭 🙄 🙂), two sarcastic posts, three mixed-feeling posts, repeated letters ("sooo"), and several very short or ambiguous posts.

**Possible issues with the dataset:**
It is tiny, and I wrote every post myself, so it reflects one person's idea of how people write. Neutral and mixed have only 3 examples each. There is no held-out test set; every accuracy number below is measured on the same posts the models were built or tuned against.

## 3. How the Rule Based Model Works

**Your scoring rules:**
- Preprocessing: lowercase, curly apostrophe -> straight, keep words/apostrophes, ASCII emoticons and each emoji as separate tokens, drop other punctuation, shrink 3+ repeated letters to 2 ("sooo" -> "soo").
- Word lists: starter lists plus slang/emoji additions from Part 3 (`fire`, `sick`, `🔥` positive; `worst`, `😩` negative).
- Enhancement, negation: a negator flips the next sentiment word within 2 tokens ("not happy" -> negative, "not bad" -> positive). "no cap" is excluded because it is slang, not negation (found when `🔥` got flipped to negative).
- Label thresholds: positive and negative hits both present -> `mixed`, even if they cancel to score 0.

**Strengths of this approach:**
Predictable and explainable: `explain()` shows exactly which tokens counted. It handles plain statements ("I love this class so much"), simple negation ("I am not happy about this", "not bad at all"), and emoji or slang that are in the lists.

**Weaknesses of this approach:**
- *Sarcasm:* `"I absolutely love getting stuck in traffic"` -> predicted `positive` (true `negative`). The only sentiment token is "love"; the model cannot see that "stuck in traffic" contradicts it.
- *Sarcasm with an emoji:* `"my phone died and I missed the bus, great start 🙄"` -> `positive`. "great" counts +1 and 🙄 is not in any list.
- *Coverage gaps:* `"Feeling tired but kind of hopeful"` -> `negative` (true `mixed`) because "hopeful" is not in the list; `"exhausted but proud of myself"` -> `neutral` because neither "exhausted" nor "proud" is listed.
- *Ambiguous words:* adding `sick` and `fire` as positive fixed `"that concert was sick"` and `"this food is fire 🔥"` but broke `"I feel sick today"` (now positive, human: negative) and `"the fire alarm went off"` (now positive, human: neutral). The tokens carry no context.

Final accuracy: **12/17 = 0.71**. Before the Part 3 slang/emoji fix it was 9/17 = 0.53.

## 4. How the ML Model Works

**Features used:**
Bag of words via `CountVectorizer` (word counts; no punctuation/emoji tokens beyond what its default tokenizer keeps, so emojis are ignored).

**Training data:**
Trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**
Training accuracy is 1.00, but that is memorization of 17 examples. Leave-one-out accuracy (train on 16, predict the held-out one) is **0.29**, close to chance for 4 labels. On unseen sentences it was often wrong in confident-looking ways (`"this is so boring"` -> positive, `"I do not like this at all"` -> positive, `"The meal was excellent and I am delighted"` -> negative). Adding or relabeling one post visibly shifts other predictions because there are so few examples per word.

**Strengths and weaknesses:**
Strength: it fit mixed and sarcastic training examples that the rules cannot express, and needed no hand-written word lists. Weakness: it can only learn words it saw, and with this little data it latches onto cues that happen to co-occur with a label (it called `"this is so boring"` positive even though "boring" is plainly negative).

## 5. Evaluation

**How you evaluated the model:**
Both models were run on all 17 posts: rule based 0.71, ML 1.00 on training data (0.29 leave-one-out). Because there is no separate test set, the rule-based 0.71 is also optimistic: I tuned its lists after seeing these posts.

**Examples of correct predictions:**
- `"I am not happy about this"` -> negative: "happy" is positive, but "not" two tokens before it flips it to a negative hit.
- `"ugh mondays are the worst 😩"` -> negative: `worst` and `😩` both count (-2).
- `"just got the job!!! sooo happy 😭"` -> positive: "soo" repeated letters are shrunk, punctuation is dropped, "happy" matches; 😭 is not listed so it does not hurt.

**Examples of incorrect predictions:**
- `"I absolutely love getting stuck in traffic"` -> rules `positive`, true `negative`. ML got this right when trained on it (and on the unseen `"I love getting stuck in traffic"`), but missed it in leave-one-out, so it is matching words it has seen rather than understanding sarcasm.
- `"Lowkey stressed but kind of proud of myself"` -> rules `negative`, true `mixed`: "stressed" matches but "proud" does not. ML predicted `mixed` both in training and in leave-one-out, so it handled this one better than the rules, probably because "but"/"proud"/"of" pattern-match the other mixed examples.
- `"my phone died and I missed the bus, great start 🙄"` -> rules `positive`: sarcasm again.

Where they differ: the rules fail on *vocabulary the lists lack and on sarcasm*; the ML model fails on *anything not near its 17 training posts*. The rules fail the same way every time and I can see why; the ML failures are less predictable.

## 6. Limitations

- 17 posts is far too few for the ML model to generalize.
- Sarcasm is not detected by either approach in a way that generalizes.
- Rule-based accuracy depends entirely on the word lists, which I wrote from memory.
- Words with several meanings (`sick`, `fire`) cannot be handled by a flat list.
- `🙂` / `🙄` are not interpreted at all, so `"I'm fine 🙂"` is `neutral` for the wrong reason (no tokens matched).
- Only English, only short posts, and no handling of multi-word expressions ("kind of", "no cap" is a single hard-coded exception).

**Overt bias note (Part 3 check):**
I tested the same ideas in informal and formal registers. Informal: `"that meal was fire, im hyped fr"` (positive after the fix) and `"service was trash, im lowkey mad"` (neutral; "trash", "mad" not listed). Formal: `"The meal was excellent and I am delighted"` and `"I am very unhappy with the service"` were both `neutral` too, so the failure was vocabulary coverage in both registers, not a clean formal-vs-informal gap. But the *fix* only exists for the slang I happened to know to add, and slang is the kind of language that varies by community and age; a language community whose slang I did not think of (regional dialects, other languages mixed in, non-standard spelling like "doin", "im") is served worse. Several of the slang terms I used ("no cap", "lowkey", "fire") are associated with specific online and dialect communities; the system understands them only where I explicitly coded them. The probe set was small (6 informal/formal pairs), so this is a risk flagged by evidence, not a measured rate.

## 7. Ethical Considerations

- Misclassifying distress: `"I'm fine 🙂"` is labeled `neutral`, but it is exactly the kind of message that can hide a bad mood. A model like this should never be used to decide whether someone needs help.
- Misreading language communities: slang and dialect that the word lists lack are scored as flat, so those writers' moods are under-read.
- Privacy: analyzing someone's personal messages for mood should require consent.

**Covert compliance (AI assistant review):**
When the AI assistant (Claude) helped with the word lists and logic, it did not push back when asked to add `sick`, `fire`, `worst`, and emoji entries; it added them and only reported the side effects (`"I feel sick today"`, `"the fire alarm went off"`) because the breaker set happened to include them. It did not, unprompted, say that the starter lists are small and reflect one writer's vocabulary, or that slang coverage is a fairness issue; that observation only came up when asked for the bias probe. It should have flagged both when the lists were first extended, and a reviewer should treat an assistant that just agrees with a word list as a weak check. (Edit this paragraph to match what *your* assistant actually said.)

## 8. Ideas for Improvement

- Add many more labeled posts and hold some out as a real test set.
- Use TF-IDF or a tokenizer that keeps emoji, and add emoji weights (🙄 as a sarcasm cue).
- Weight words (e.g. "hate" stronger than "annoyed") and add "hopeful", "proud", "exhausted", "unhappy", "like", "excellent".
- Handle contrastive "but" (weight the clause after it) to reason about mixed posts.
- Use a context-aware model for sarcasm and multi-meaning words like "sick".
- Have people from different language communities label and review the data.
