import pandas as pd
import numpy as np
import re

countline = 0


def subsequence(seq):
    offset0 = re.findall(r'...', seq)
    offset1 = re.findall(r'...', seq[1:])
    offset2 = re.findall(r'...', seq[2:])
    return [offset0, offset1, offset2]


def sequences2subsequences(seq):
    return list(map(subsequence, seq))


def seq2fixed_length_vec(seqs, vocabulary):
    flat_threemers = [item for sublist in seqs for item in sublist]
    vec = np.array([0.0] * 100)
    for t in flat_threemers:
        try:
            vec = np.add(vec, vocabulary[t])
        except:
            vec = np.add(vec, vocabulary['<unk>'])
    global countline
    countline += 1
    if countline % 10 == 0:
        print(countline)

    return vec


def seq2wordcount_vec(seqs, vocabulary):
    flat_threemers = [item for sublist in seqs for item in sublist]
    wordcount = []
    for val in range(len(vocabulary.keys())):
        wordcount.append([0.0] * 100)
    for t in flat_threemers:
        try:
            wordcount[list(vocabulary.keys()).index(t)] = np.add(wordcount[list(vocabulary.keys()).index(t)],
                                                                 vocabulary[t])
        except:
            wordcount[list(vocabulary.keys()).index('<unk>')] = np.add(
                wordcount[list(vocabulary.keys()).index('<unk>')],
                vocabulary['<unk>'])
    return np.array(wordcount)


def seq2fixed_vec_matrix(seqs, vocabulary):
    flat_threemers = [item for sublist in seqs for item in sublist]
    vec_array = []
    for t in flat_threemers:
        try:
            vec_array.append(vocabulary[t])
        except:
            vec_array.append(vocabulary['<unk>'])
    while len(vec_array) < 698:
        vec_array.append(np.array([0.0] * 100))
    return np.array(vec_array)


def handle_ec_number(ec_number, ecn_level):
    if ecn_level <= 0 or ecn_level > 4:
        raise ValueError('EC number level must be between 1-4')
    keep = True
    numbers = ec_number.split(';')
    ret = numbers[0].split('.')[:ecn_level]
    for n in numbers:
        n = n.split('.')[:ecn_level]
        if ret != n:
            keep = False
    if keep:
        return ','.join(ret)
    else:
        return None


#  Read datasets
data = pd.read_csv("ecpred_uniprot_uniref_90.csv").head(500)
embeddings = pd.read_csv("protVec_100d_3grams.csv", sep='\\t', engine='python', header=None)

'''
firstdig = {}
for row in data['ec_number']:
    n = row.split('.')
    if n[0] not in firstdig:
        firstdig[n[0]] = 0
    else:
        firstdig[n[0]] += 1

print(firstdig)
5/0
'''


def parse_dataset(dataset, vocabulary, representation='matrix', maxlen=700, ecn_level=1):
    dataset = dataset[dataset.get('sequence').notna()]  # Removing null values from sequence column
    dataset = dataset[dataset['sequence'].apply(lambda x: len(x) < maxlen)]  # Removing sequences with length > 1500
    dataset['subsequences'] = sequences2subsequences(dataset.get('sequence'))

    if representation == "matrix":
        matrix = dataset['subsequences'].apply(lambda x: seq2fixed_vec_matrix(x, vocabulary))
        dataset['vectors'] = matrix
        dataset = dataset[["ec_number", "vectors"]]
    elif representation == 'vocabulary':
        matrix = dataset['subsequences'].apply(lambda x: seq2wordcount_vec(x, vocabulary))
        dataset['vectors'] = matrix
        dataset = dataset[["ec_number", "vectors"]]
    elif representation == 'vector':
        one_d_vecs = dataset['subsequences'].apply(lambda x: seq2fixed_length_vec(x, vocabulary))
        dataset = dataset[["ec_number"]]
        dataset = dataset.join(one_d_vecs)
    else:
        raise ValueError('Insert valid representation: "vector", "matrix" or "vocabulary"')

    dataset['ec_number'] = dataset['ec_number'].apply(lambda x: handle_ec_number(x, ecn_level))
    dataset = dataset[dataset.get('ec_number').notna()]
    return dataset


#  Build threemer dictionary
embeddings[100] = embeddings[100].apply(lambda elem: elem[:-1])
threemers = embeddings.get(0)
vocab = {}
for i, kmer in enumerate(threemers):
    vocab[kmer[1:]] = embeddings.iloc[i][1:].to_numpy(dtype="float32")

df = parse_dataset(data, vocab, ecn_level=2, representation='matrix')

pd.to_pickle(df, 'parsed_data/data-micro.pkl')

