import random
import data_helpers as dhrt
from tensorflow.contrib import learn
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from itertools import product


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, LSTM, Bidirectional, BatchNormalization, GlobalAveragePooling1D, Input, Reshape, Concatenate, concatenate
from keras.optimizers import SGD, Adam
from keras.layers.merge import Dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from keras.preprocessing.sequence import skipgrams
from keras.utils import np_utils
import matplotlib.pyplot as plt
from Bio import pairwise2


# Network Parameters
learning_rate = 0.001
num_classes = 2
num_features = 372
word_length = 4
batch_size = 4
nb_epoch = 16
hidden_size = 100
num_sequences = 10
steps_per_epoch = 10
num_classes = 2
num_filters = [16, 4]


def replace_spaces(x):
    return x.replace(' ', '')


def get_alignments(x, y, seq_i, seq_j, batch_size):
    '''
    Aligns every pair of sequences to prepare input to CNN
    :param x: a set of sequences
    :param y: a set of labels
    :return: a set of pairwise alignments of the sets in x (cartesian product) x
    '''
    a = pairwise2.align.localxx(x[seq_i], x[seq_j], one_alignment_only=True)[0]
    align_x = np.array(list(a)[0:2])
    if np.array_equal(y[seq_i], y[seq_j]):
        align_y = np.array([1])
    else:
        align_y = np.array([0])
    for i in range(seq_i + 1, seq_i + batch_size):
        for j in range(seq_j + 1, seq_j + batch_size):
            a = pairwise2.align.localxx(x[i], x[j], one_alignment_only=True)[0]
            align_x = np.vstack((align_x, np.array(list(a)[0:2])))
            if np.array_equal(y[i], y[j]):
                align_y = np.append(align_y, np.array([1]))
            else:
                align_y = np.append(align_y, np.array([0]))
    lens = [len(seq) for alignment in align_x for seq in alignment]
    max_document_length = max(lens)
    return align_x, align_y, max_document_length


def get_vocab(chars):
    vocab = {}
    i = 0
    words = product(chars, repeat=word_length)
    word_list = []  # Create a empty list
    for permutation in words:
        word_list.append(''.join(permutation))  # Join alphabet together and append in namList
    for word in word_list:
        vocab[word] = i
        i += 1
    print('Vocab:', vocab)
    return vocab


def base_pairs_to_onehot(seq1, seq2, max_len):
    vocab = get_vocab('atcgx')
    index_arr = np.array([])
    for i in range(0, max_len):
        if i < len(seq1):
            index_arr = np.append(index_arr, vocab[seq1[i]+seq2[i]])
        else:
            index_arr = np.append(index_arr, 6)
    return index_arr


def split_alignments(x, max_len):
    s1 = []
    s2 = []
    for i in range(len(x)):
        proc1 = ''
        proc2 = ''
        for j in range(max_len):
            #print(i, j)
            if j < len(x[i][0]):
                proc1 += (x[i][0][j]).replace('-', 'x')
                proc2 += (x[i][1][j]).replace('-', 'x')
            else:
                proc1 += 'x'
                proc2 += 'x'
                #re.findall('..', '1234567890')
        #proc = [proc[i:i+word_length] for i in range(0, len(proc), word_length)]
        proc1 = ' '.join([proc1[i:i + word_length] for i in range(0, len(proc1), word_length)])
        proc2 = ' '.join([proc2[i:i + word_length] for i in range(0, len(proc2), word_length)])
        #proc = re.findall('....', proc)
        #print(proc)
        s1 = np.append(s1, proc1)
        s2 = np.append(s2, proc2)
    return s1, s2


def zip_alignments(x, max_len):
    x_proc = []
    for i in range(len(x)):
        proc = ''
        for j in range(max_len):
            #print(i, j)
            if j < len(x[i][0]):
                proc += (x[i][0][j]+x[i][1][j]).replace('-', 'x')
            else:
                proc += 'xx'
                #re.findall('..', '1234567890')
        #proc = [proc[i:i+word_length] for i in range(0, len(proc), word_length)]
        proc = ' '.join([proc[i:i + word_length] for i in range(0, len(proc), word_length)])
        #proc = re.findall('....', proc)
        #print(proc)
        x_proc = np.append(x_proc, proc)
    return x_proc


def generate_vec_batch(x_train, y_train, batch_size, tokenizer, SkipGram):
    indices_i = np.arange(0, len(x_train) - 1 - batch_size)
    indices_j = np.arange(0, len(x_train) - 1 - batch_size)
    while True:
        if len(indices_i) == 0:
            indices_i = np.arange(0, len(x_train) - 1 - batch_size)
        if len(indices_j) == 0:
            indices_j = np.arange(0, len(x_train) - 1 - batch_size)
        i = np.random.choice(indices_i, 1, replace=False)[0]
        j = np.random.choice(indices_j, 1, replace=False)[0]
        if (i + batch_size) < len(x_train) and (j + batch_size) < len(x_train):
            align_x, align_y, max_length = (get_alignments(x_train, y_train, i, j, batch_size))
        elif (i + batch_size) >= len(x_train):
            print('End of training set, temp batch:', len(x_train[i:]))
            align_x, align_y, max_length = (get_alignments(x_train, y_train, i, j, len(x_train[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, max_length = (get_alignments(x_train, y_train, i, j, len(x_train[j:])))
        #text = zip_alignments(align_x, max_length)
        s1, s2 = split_alignments(align_x, max_length)
        for _, doc in enumerate(pad_sequences(tokenizer.texts_to_sequences(np.append(s1, s2)), maxlen=max_length, padding='post')):
            data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=word_length, negative_samples=5.)
            x = [np.array(x) for x in zip(*data)]
            y = np.array(labels, dtype=np.int32)
            if x:
                yield x, y


def alignments2vec(x, y, V, tokenizer):
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(V, word_length)(w_inputs)

    # context
    c_inputs = Input(shape=(1,), dtype='int32')
    c = Embedding(V, word_length)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)

    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.summary()
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')

    history = SkipGram.fit_generator(generate_vec_batch(x, y, batch_size, tokenizer, SkipGram),
                           steps_per_epoch=steps_per_epoch,
                           epochs=len(x_train)//batch_size//steps_per_epoch,
                           validation_data=generate_vec_batch(x, y, batch_size, tokenizer, SkipGram),
                           validation_steps=steps_per_epoch,
                           verbose=1)

    print(history.history.keys())
    # summarize history for accuracy
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    f = open('alignment_vec.txt', 'w')
    f.write('{} {}\n'.format(V - 1, word_length))
    vectors = SkipGram.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()

    w2v = gensim.models.KeyedVectors.load_word2vec_format('./alignment_vec.txt', binary=False)
    print(w2v.most_similar(positive=['aaaa']))



def convert_base_pairs(x, y, max_document_length):
    x_proc = []
    y_proc = []
    for i in range(len(x)):
        proc = base_pairs_to_onehot(x[i][0], x[i][1], max_document_length)
        if len(x_proc) == 0:
            x_proc = proc
            y_proc = y[i]
        else:
            x_proc = np.vstack((x_proc, proc))
            y_proc = np.append(y_proc, y[i])
    x_proc = np.reshape(x_proc, (y_proc.shape[0], -1))
    return x_proc, y_proc, max_document_length


def get_list_of_word2vec(x, w2v, max_length, n_samples):
    word_vec = []
    for alignment in x:
        vec = []
        word_list = alignment.split(' ')
        if len(word_list[-1]) != word_length:
            word_list = word_list[:-1]
        for word in word_list:
            try:
                if len(vec) == 0:
                    vec = w2v.word_vec(word)
                else:
                    vec = np.vstack((vec, w2v.word_vec(word)))
            except Exception as e:
                print('Word', word, 'not in vocab')
        vec = np.reshape(vec, (len(word_list), -1))
        if len(word_vec) == 0:
            word_vec = vec
        else:
            word_vec = np.dstack((word_vec, vec))
    word_vec = np.reshape(word_vec, (n_samples, -1, word_length))
    return word_vec


def generate_word2vec_batch(x, y):
    indices_i = np.arange(0, len(x) - 1 - batch_size)
    indices_j = np.arange(0, len(x) - 1 - batch_size)
    while True:
        if len(indices_i) == 0:
            indices_i = np.arange(0, len(x) - 1 - batch_size)
        if len(indices_j) == 0:
            indices_j = np.arange(0, len(x) - 1 - batch_size)
        i = np.random.choice(indices_i, 1, replace=False)[0]
        j = np.random.choice(indices_j, 1, replace=False)[0]
        w2v = gensim.models.KeyedVectors.load_word2vec_format('./alignment_vec.txt', binary=False)
        if (i + batch_size) < len(x) and (j + batch_size) < len(x):
            align_x, align_y, max_length = (get_alignments(x, y, i, j, batch_size))
        elif (i + batch_size) >= len(x):
            print('End of training set, temp batch:', len(x[i:]))
            align_x, align_y, max_length = (get_alignments(x, y, i, j, len(x[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, max_length = (get_alignments(x, y, i, j, len(x[j:])))
        text = zip_alignments(align_x, max_length)
        s1, s2 = split_alignments(align_x, max_length)
        word2vec1 = get_list_of_word2vec(s1, w2v, max_length, align_y.shape[0])
        word2vec2 = get_list_of_word2vec(s2, w2v, max_length, align_y.shape[0])
        align_y = np_utils.to_categorical(align_y)
        if align_y.shape[1] == 2:
            yield [word2vec1, word2vec2], align_y


def generate_batch(x, y):
    indices_i = np.arange(0, len(x) - 1 - batch_size)
    indices_j = np.arange(0, len(x) - 1 - batch_size)
    while True:
        if len(indices_i) == 0:
            indices_i = np.arange(0, len(x) - 1 - batch_size)
        if len(indices_j) == 0:
            indices_j = np.arange(0, len(x) - 1 - batch_size)
        i = np.random.choice(indices_i, 1, replace=False)[0]
        j = np.random.choice(indices_j, 1, replace=False)[0]
        if (i + batch_size) < len(x) and (j + batch_size) < len(x):
            align_x, align_y, _ = (get_alignments(x, y, i, j, batch_size))
        elif (i + batch_size) >= len(x):
            print('End of training set, temp batch:', len(x[i:]))
            align_x, align_y, _ = (get_alignments(x, y, i, j, len(x[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, _ = (get_alignments(x, y, i, j, len(x[j:])))
        align_x, align_y = convert_base_pairs(align_x, align_y)
        yield align_x, align_y


# load data
x_rt, y_rt = dhrt.load_data_and_labels('h3.pos', 'h3.neg')

x_rt = np.array([replace_spaces(seq) for seq in x_rt])
print('X:', x_rt)
y_rt = np.array(list(y_rt))
print(pairwise2.align.localxx('aaattcgctgc','aaatctcgcgat', one_alignment_only=True))
shuffled_rt = np.random.permutation(range(len(x_rt)))
x_shuffle = x_rt[shuffled_rt]
y_shuffle = y_rt[shuffled_rt]

x_train, x_valid, y_train, y_valid = train_test_split(x_shuffle,
                                                      y_shuffle,
                                                      stratify=y_shuffle,
                                                      test_size=0.2)

print('x shape:', x_train.shape)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(get_vocab('atcgx'))
V = len(tokenizer.word_index) + 1
print('Num Words:', V)

#alignments2vec(x_train, y_train, V, tokenizer) #uncomment to train word2vec representation
'''
model = Sequential()
'''
'''
model.add(Conv1D(filters=64, kernel_size=word_length, input_shape=(None, word_length)))
model.add(Activation('relu'))
model.add(Conv1D(filters=64, kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
'''
model.add(LSTM(32, return_sequences=True, stateful=True, batch_input_shape=((batch_size * batch_size - 2 * batch_size + 2), None, word_length)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(num_classes, activation='softmax'))
'''
encoder_a = Input(shape=(None, word_length))
conv_1_a = Conv1D(64, word_length, activation='relu')(encoder_a)
conv_2_a = Conv1D(64, 3, activation='relu')(conv_1_a)
pool_a = MaxPooling1D(3)(conv_2_a)
conv_3_a = Conv1D(128, 3, activation='relu')(pool_a)
conv_4_a = Conv1D(128, 3, activation='relu')(conv_3_a)
lstm_a = LSTM(32)(conv_4_a)
dense_a = Dense(32, activation='relu')(lstm_a)

encoder_b = Input(shape=(None, word_length))
conv_1_b = Conv1D(64, word_length, activation='relu')(encoder_b)
conv_2_b = Conv1D(64, 3, activation='relu')(conv_1_b)
pool_b = MaxPooling1D(3)(conv_2_b)
conv_3_b = Conv1D(128, 3, activation='relu')(pool_b)
conv_4_b = Conv1D(128, 3, activation='relu')(conv_3_b)
lstm_b = LSTM(32)(conv_3_b)
dense_b = Dense(32, activation='relu')(lstm_b)

decoder = concatenate([dense_a, dense_b])

dense = Dense(32, activation='relu')(decoder)
output = Dense(num_classes, activation='softmax')(dense)
model = Model(inputs=[encoder_a, encoder_b], outputs=output)

adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Training shapes:', x_train.shape, y_train.shape)
print('Valid shapes:', x_valid.shape, y_valid.shape)
print(model.summary())
'''
history = model.fit_generator(generate_batch(x_train, y_train),
                              steps_per_epoch=10,
                              epochs=10,#len(x_train)//batch_size//10,
                              validation_data=generate_batch(x_valid, y_valid),
                              validation_steps=10,
                              verbose=1)
'''
history = model.fit_generator(generate_word2vec_batch(x_train, y_train),
                              steps_per_epoch=steps_per_epoch,
                              epochs=16,#len(x_train)//batch_size//steps_per_epoch,
                              validation_data=generate_word2vec_batch(x_valid, y_valid),
                              validation_steps=steps_per_epoch,
                              verbose=1)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
