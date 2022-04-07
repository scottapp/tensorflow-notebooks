import io
import pandas as pd
import pickle


def load_pickle(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
        return obj


def save_pickle(obj, filename, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, protocol)


def load_ff_data():
    with open('../data/F-F_Research_Data_Factors.csv', 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = lines[3:1151]
    data = io.StringIO(''.join(lines))
    df = pd.read_csv(data)
    df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df.index = df.index + pd.offsets.MonthEnd()
    df = df.apply(lambda x: x / 100)
    return df
