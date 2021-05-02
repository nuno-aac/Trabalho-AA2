import pandas as pd
import numpy as np
import re


def subsequence(seq):
    offset0 = re.findall(r'...', seq)
    offset1 = re.findall(r'...', seq[1:])
    offset2 = re.findall(r'...', seq[2:])
    return [offset0, offset1, offset2]


def sequences2subsequences(seq):
    return list(map(subsequence, seq))


def seq2fixed_length_vec(seqs):
    flat_threemers = np.array(seqs, dtype=object).flatten()
    vec = np.array([0.0] * 100)
    for t in flat_threemers:
        try:
            vec = np.add(vec, embeddings.iloc[threemersidx[t]])
        except:
            vec = np.add(vec, embeddings.iloc[threemersidx['<unk>']])
    return vec


def seq2wordcount_vec(seqs):
    flat_threemers = np.array(seqs, dtype=object).flatten()
    wordcount = {}
    for key in threemersidx:
        wordcount[key] = [0.0] * 100
    for t in flat_threemers:
        try:
            wordcount[t] = np.add(wordcount[t], embeddings.iloc[threemersidx[t]])
        except:
            wordcount['<unk>'] = np.add(wordcount['<unk>'], embeddings.iloc[threemersidx['<unk>']])
    return wordcount


def seq2fixed_vec_matrix(seqs):
    flat_threemers = np.array(seqs, dtype=object).flatten()
    vec_array = []
    for t in flat_threemers:
        try:
            vec_array.append(embeddings.iloc[threemersidx[t]])
        except:
            vec_array.append(embeddings.iloc[threemersidx['<unk>']])
    while len(vec_array) <= 1498:
        vec_array.append([0.0]*100)
    return vec_array


#  Read datasets
data = pd.read_csv("ecpred_uniprot_uniref_90.csv").head(50)
embeddings = pd.read_csv("protVec_100d_3grams.csv", sep='\\t', engine='python', header=None)

#  Build threemer dictionary
threemers = embeddings.get(0)
threemersidx = {}  # generate word to index translation dictionary. Use for kmersdict function arguments.
for i, kmer in enumerate(threemers):
    threemersidx[kmer[1:]] = i
# print(threemersidx)

#  Build embedding matrix
embeddings[100] = embeddings[100].apply(lambda elem: elem[:-1])  # Remover aspas
del embeddings[0]
embeddings = embeddings.astype(float)
print(embeddings)

data = data[data.get('sequence').notna()]  # Removing null values from sequence column
sequences = data.get('sequence')

subsequences = sequences2subsequences(sequences)

data['subsequences'] = subsequences

print(data)

print(data['subsequences'].apply(seq2fixed_length_vec))
print(data['subsequences'].apply(seq2wordcount_vec))
