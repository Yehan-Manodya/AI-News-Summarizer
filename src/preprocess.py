import re
import nltk
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Special tokens every NLP model needs
PAD_TOKEN = '<PAD>'   # padding to make sequences equal length
SOS_TOKEN = '<SOS>'   # Start Of Sequence - decoder needs this to start generating
EOS_TOKEN = '<EOS>'   # End Of Sequence - model knows when to stop
UNK_TOKEN = '<UNK>'   # Unknown words not in vocabulary

class Vocabulary:
    def __init__(self):
        # word - number
        self.word2idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3
        }
        # number - word (for converting predictions back to text)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = Counter()

    def build(self, texts, max_vocab_size=20000):
        """Build vocabulary from a list of texts"""
        for text in texts:
            tokens = nltk.word_tokenize(text.lower())
            self.word_count.update(tokens)

        # Keep only the most common words
        most_common = self.word_count.most_common(max_vocab_size - 4)  # -4 for special tokens

        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"✅ Vocabulary built: {len(self.word2idx):,} words")

    def encode(self, text, max_len, add_eos=False):
    # Leave room for EOS token if needed
        actual_max = max_len - 1 if add_eos else max_len
        tokens = nltk.word_tokenize(text.lower())[:actual_max]
        ids = [self.word2idx.get(t, self.word2idx[UNK_TOKEN]) for t in tokens]

        if add_eos:
            ids = ids + [self.word2idx[EOS_TOKEN]]

    # Pad to max_len
        ids = ids + [self.word2idx[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        """Convert list of numbers → text string"""
        words = []
        for idx in ids:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if word in [PAD_TOKEN, EOS_TOKEN, SOS_TOKEN]:
                break
            words.append(word)
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)


def clean_text(text):
    """Clean raw text"""
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'\(CNN\)', '', text)                # remove CNN artifact
    text = re.sub(r'-- ', '', text)                    # remove dashes
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"-]', '', text)  # special chars
    text = re.sub(r'\s+', ' ', text).strip()           # extra spaces
    return text


def prepare_data(df, vocab=None, article_max_len=400, summary_max_len=80):
    """
    Takes a dataframe with 'article' and 'summary' columns
    Returns encoded articles, encoded summaries, and vocabulary
    """
    # Clean
    df['article'] = df['article'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_text)

    # Build vocab if not provided
    if vocab is None:
        vocab = Vocabulary()
        all_texts = list(df['article']) + list(df['summary'])
        vocab.build(all_texts)

    # Encode
    encoded_articles  = [vocab.encode(a, article_max_len) for a in df['article']]
    encoded_summaries = [vocab.encode(s, summary_max_len, add_eos=True) for s in df['summary']]

    return encoded_articles, encoded_summaries, vocab