import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import sequence
from keras.preprocessing import text
import tensorflow as tf
import pickle
from preprocessor import TextPreprocessor


def load_data(train_file, eval_file):
    encoder = OneHotEncoder()

    train_set = pd.read_csv('data/{}'.format(train_file), sep='\t')
    train_set.columns = ['label', 'title']
    train_set = shuffle(train_set, random_state=22)
    train_hot = encoder.fit_transform(train_set[['label']]).toarray()

    eval_set = pd.read_csv('data/{}'.format(eval_file), sep='\t')
    eval_set.columns = ['label', 'title']
    eval_hot = encoder.transform(eval_set[['label']]).toarray()

    return (train_set['title'].values, train_hot), (eval_set['title'].values, eval_hot)


if __name__ == '__main__':
    train = pd.read_csv('data/train.tsv', sep='\t')
    print(train.shape)
    train = shuffle(train, random_state=42)

    (train_texts, train_labels), (eval_texts, eval_labels) = load_data('train.tsv', 'eval.tsv')
    print('text: %s' % train_texts[0])
    print('label: %s' % train_labels[0])
    print(len(train_texts), len(train_labels))

    print('text: %s' % eval_texts[0])
    print('label: %s' % eval_labels[0])
    print(len(eval_texts), len(eval_labels))

    VOCAB_SIZE = 20000
    MAX_SEQUENCE_LENGTH = 50

    ((train_texts, train_labels), (eval_texts, eval_labels)) = load_data('train.tsv', 'eval.tsv')

    # Create vocabulary from training corpus.
    processor = TextPreprocessor(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    processor.fit(train_texts)

    # Preprocess the data
    train_texts_vectorized = processor.transform(train_texts)
    eval_texts_vectorized = processor.transform(eval_texts)

    path = 'processor_state.pkl'
    with open(path, 'wb') as f:
        pickle.dump(processor, f)

    print('saved pkl at %s' % path)
