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
    flat_threemers = [j for n in seqs for j in n]
    vec = [0] * 100
    for t in flat_threemers:
        try:
            vec = np.add(vec, embeddings.iloc[threemersidx[kmer]].apply(float))
        except:
            vec = np.add(vec, embeddings.iloc[threemersidx['<unk>']].apply(float))
    return vec


data = pd.read_csv("ecpred_uniprot_uniref_90.csv")
embeddings = pd.read_csv("protVec_100d_3grams.csv", sep='\\t', engine='python', header=None)

#  Build threemer dictionary
threemers = embeddings.get(0)
threemersidx = {}  # generate word to index translation dictionary. Use for kmersdict function arguments.
for i, kmer in enumerate(threemers):
    threemersidx[kmer[1:]] = i
print(threemersidx)

#  Build embedding matrix
embeddings[100] = embeddings[100].apply(lambda elem: elem[:-1])  # Remover aspas
del embeddings[0]
print(embeddings)

data = data[data.get('sequence').notna()]  # Removing null values from sequence column
sequences = data.get('sequence')

maxi = 0
for sequen in sequences:
    if len(sequen) > 1500:
        maxi += 1
print("max sequence length: " + str(maxi))

subsequences = sequences2subsequences(sequences)

data['subsequences'] = subsequences

print(data)

#  print(data['subsequences'].apply(seq2fixed_length_vec))
