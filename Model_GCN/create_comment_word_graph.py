import os
import random
import nltk
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

dataset = 'Amazon'
word_embeddings_dim = 300
word_vector_map = {}
comment_name_list = []
comment_train_list = []
comment_test_list = []

f = open('data/' + 'Amazon.txt', 'r')
lines = f.readlines()
for line in lines:
    comment_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        comment_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        comment_train_list.append(line.strip())
f.close()

comment_content_list = []
f = open('data/corpus/' + 'Amazon.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    comment_content_list.append(line.strip())
f.close()

train_ids = []
for train_name in comment_train_list:
    train_id = comment_name_list.index(train_name)
    train_ids.append(train_id)
print(' \n train_ids \n')
print(train_ids)
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + 'Amazon.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in comment_test_list:
    test_id = comment_name_list.index(test_name)
    test_ids.append(test_id)
print(' \n test_ids \n')
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + 'Amazon.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids

shuffle_comment_name_list = []
shuffle_comment_words_list = []
for id in ids:
    shuffle_comment_name_list.append(comment_name_list[int(id)])
    shuffle_comment_words_list.append(comment_content_list[int(id)])
shuffle_comment_name_str = '\n'.join(shuffle_comment_name_list)
shuffle_comment_words_str = '\n'.join(shuffle_comment_words_list)

f = open('data/' + 'Amazon_shuffle.txt', 'w')
f.write(shuffle_comment_name_str)
f.close()

f = open('data/corpus/' + 'Amazon_shuffle.txt', 'w')
f.write(shuffle_comment_words_str)
f.close()

word_freq = {}
word_set = set()
for comment_words in shuffle_comment_words_list:
    words = comment_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_comment_list = {}


for i in range(len(shuffle_comment_words_list)):
    comment = shuffle_comment_words_list[i]
    words = comment.split()
    visited = set()
    for word in words:
        if word in visited:
            continue
        if word in word_comment_list:
            doc_list = word_comment_list[word]
            doc_list.append(i)
            word_comment_list[word] = doc_list
        else:
            word_comment_list[word] = [i]
        visited.add(word)


word_comment_freq = {}
for word, comment_list in word_comment_list.items():
    word_comment_freq[word] = len(comment_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + 'Amazon_vocab.txt', 'w')
f.write(vocab_str)
f.close()

definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)
f = open('data/corpus/' + 'Amazon_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)
f = open('data/corpus/' + 'Amazon_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + 'Amazon_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])


label_set = set()
for doc_meta in shuffle_comment_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + 'Amazon_labels.txt', 'w')
f.write(label_list_str)
f.close()


train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size


real_train_comment_names = shuffle_comment_name_list[:real_train_size]
real_train_comment_names_str = '\n'.join(real_train_comment_names)

f = open('data/' + 'Amazon.real_train.name', 'w')
f.write(real_train_comment_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    comment_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    comment_words = shuffle_comment_words_list[i]
    words = comment_words.split()
    comment_len = len(words)
    if comment_len<1:
        print('comment_len::'+str(comment_len))
        print(comment_words)
#        comment_len=1;
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            comment_vec = comment_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        data_x.append(comment_vec[j] / comment_len)

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    comment_meta = shuffle_comment_name_list[i]
    temp = comment_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
print(y)

test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    comment_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    comment_words = shuffle_comment_words_list[i + train_size]
    words = comment_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            comment_vec = comment_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        data_tx.append(comment_vec[j] / comment_len)

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    comment_meta = shuffle_comment_name_list[i + train_size]
    temp = comment_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
print(ty)


word_vectors = np.random.uniform(-0.01, 0.01,(vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    comment_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    comment_words = shuffle_comment_words_list[i]
    words = comment_words.split()
    comment_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            comment_vec = comment_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        data_allx.append(comment_vec[j] / doc_len)
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_comment_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

window_size = 20
windows = []

for comment_words in shuffle_comment_words_list:
    words = comment_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

word_window_freq = {}
for window in windows:
    visited = set()
    for i in range(len(window)):
        if window[i] in visited:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        visited.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []


num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# comment word frequency
comment_word_freq = {}

for comment_id in range(len(shuffle_comment_words_list)):
    comment_words = shuffle_comment_words_list[comment_id]
    words = comment_words.split()
    for word in words:
        word_id = word_id_map[word]
        comment_word_str = str(comment_id) + ',' + str(word_id)
        if comment_word_str in comment_word_freq:
            comment_word_freq[comment_word_str] += 1
        else:
            comment_word_freq[comment_word_str] = 1

for i in range(len(shuffle_comment_words_list)):
    comment_words = shuffle_comment_words_list[i]
    words = comment_words.split()
    comment_word_set = set()
    for word in words:
        if word in comment_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = comment_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_comment_words_list) /
                  word_comment_freq[vocab[j]])
        weight.append(freq * idf)
        comment_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()