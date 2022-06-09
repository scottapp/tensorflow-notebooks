from keras.preprocessing import sequence
from keras.preprocessing import text


class TextPreprocessor(object):
    def __init__(self, vocab_size, max_sequence_length):
        self._vocab_size = vocab_size
        self._max_sequence_length = max_sequence_length
        self._tokenizer = None

    def fit(self, text_list):
        # Create vocabulary from input corpus.
        tokenizer = text.Tokenizer(num_words=self._vocab_size)
        tokenizer.fit_on_texts(text_list)
        self._tokenizer = tokenizer

    def transform(self, text_list):
        # Transform text to sequence of integers
        text_sequence = self._tokenizer.texts_to_sequences(text_list)

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        padded_text_sequence = sequence.pad_sequences(text_sequence, maxlen=self._max_sequence_length)
        return padded_text_sequence
