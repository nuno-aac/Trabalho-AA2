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


data = pd.read_csv("ecpred_uniprot_uniref_90.csv").head(20)
embeddings = pd.read_csv("protVec_100d_3grams.csv", sep='\\t', engine='python', header=None)

threemers = embeddings.get(0)

threemersidx = {}  # generate word to index translation dictionary. Use for kmersdict function arguments.
for i, kmer in enumerate(threemers):
    threemersidx[kmer[1:]] = i

print(threemersidx)

data = data[data.get('sequence').notna()]  # Removing null values from sequence column
sequences = data.get('sequence')

subsequences = sequences2subsequences(sequences)

data['subsequences'] = subsequences

print(data)
