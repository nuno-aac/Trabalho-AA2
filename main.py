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
    flat_threemers = [item for sublist in seqs for item in sublist]
    vec = np.array([0.0] * 100)
    for t in flat_threemers:
        try:
            vec = np.add(vec, embeddings.iloc[threemersidx[t]])
        except:
            vec = np.add(vec, embeddings.iloc[threemersidx['<unk>']])
    return vec


def seq2wordcount_vec(seqs):
    flat_threemers = [item for sublist in seqs for item in sublist]
    wordcount = []
    for val in range(len(threemersidx)):
        wordcount.append([0.0] * 100)
    for t in flat_threemers:
        try:
            wordcount[threemersidx[t]] = np.add(wordcount[threemersidx[t]], embeddings.iloc[threemersidx[t]])
        except:
            wordcount[threemersidx['<unk>']] = np.add(wordcount[threemersidx['<unk>']], embeddings.iloc[threemersidx['<unk>']])
    return np.array(wordcount)


def seq2fixed_vec_matrix(seqs):
    flat_threemers = [item for sublist in seqs for item in sublist]
    vec_array = []
    for t in flat_threemers:
        try:
            vec_array.append(embeddings.iloc[threemersidx[t]].to_numpy())
        except:
            vec_array.append(embeddings.iloc[threemersidx['<unk>']].to_numpy())
    while len(vec_array) < 698:
        vec_array.append(np.array([0.0]*100))
    return np.array(vec_array)


#  Read datasets
data = pd.read_csv("ecpred_uniprot_uniref_90.csv").head(11000)
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
data = data[data['sequence'].apply(lambda x: len(x) < 700)]  # Removing sequences with length > 1500
sequences = data.get('sequence')

subsequences = sequences2subsequences(sequences)

data['subsequences'] = subsequences

# print(data['subsequences'].apply(seq2fixed_length_vec))
# print(data['subsequences'].apply(seq2wordcount_vec))
# print(data['subsequences'].apply(seq2fixed_vec_matrix))
data['vectors'] = data['subsequences'].apply(seq2fixed_vec_matrix)

print(data.iloc[0].get('vectors'))

data = data[["uniref_90", "ec_number", "vectors"]]

pd.to_pickle(data, './data.pkl')


