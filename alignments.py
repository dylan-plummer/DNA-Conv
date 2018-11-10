import random
import data_helpers as dhrt
from tensorflow.contrib import learn
import numpy as np
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, Conv1D, AveragePooling1D, LSTM, Bidirectional, BatchNormalization, GlobalAveragePooling1D, Input, Reshape
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
batch_size = 8
nb_epoch = 16
hidden_size = 100
num_sequences = 10
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
    for c1 in chars:
        for c2 in chars:
            vocab[c1+c2] = i
            i += 1
    return vocab


def base_pairs_to_onehot(seq1, seq2, max_len):
    vocab = get_vocab('atcg-')
    index_arr = np.array([])
    for i in range(0, max_len):
        if i < len(seq1):
            index_arr = np.append(index_arr, vocab[seq1[i]+seq2[i]])
        else:
            index_arr = np.append(index_arr, 6)
    return index_arr


def zip_alignments(x, max_len):
    x_proc = []
    for i in range(len(x)):
        proc = ''
        for j in range(max_len):
            if j < len(x[i][0]):
                proc += x[i][0][j]+x[i][1][j] + ' '
            else:
                proc += '-- '
        x_proc = np.append(x_proc, proc.replace('-', 'x'))
    return x_proc


def generate_vec_batch(x_train, y_train, batch_size, tokenizer, SkipGram):
    while True:
        i = random.randint(0, len(x_train) - 1)
        j = random.randint(0, len(x_train) - 1)
        if (i + batch_size) < len(x_train) and (j + batch_size) < len(x_train):
            align_x, align_y, max_length = (get_alignments(x_train, y_train, i, j, batch_size))
        elif (i + batch_size) >= len(x_train):
            print('End of training set, temp batch:', len(x_train[i:]))
            align_x, align_y, max_length = (get_alignments(x_train, y_train, i, j, len(x_train[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, max_length = (get_alignments(x_train, y_train, i, j, len(x_train[j:])))
        text = zip_alignments(align_x, max_length)
        print(text)
        for _, doc in enumerate(pad_sequences(tokenizer.texts_to_sequences(text), maxlen=max_length, padding='post')):
            data, labels = skipgrams(sequence=doc, vocabulary_size=V, window_size=5, negative_samples=5.)
            x = [np.array(x) for x in zip(*data)]
            y = np.array(labels, dtype=np.int32)
            if x:
                yield x, y


def alignments2vec(x, y, V, tokenizer):
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(V, num_features)(w_inputs)

    # context
    c_inputs = Input(shape=(1,), dtype='int32')
    c = Embedding(V, num_features)(c_inputs)
    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)
    o = Activation('sigmoid')(o)

    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.summary()
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')

    SkipGram.fit_generator(generate_vec_batch(x, y, batch_size, tokenizer, SkipGram),
                           steps_per_epoch=10,
                           epochs=10,#len(x_train)//batch_size//10,
                           validation_data=generate_vec_batch(x, y, batch_size, tokenizer, SkipGram),
                           validation_steps=10,
                           verbose=1)

    f = open('alignment_vec.txt', 'w')
    f.write('{} {}\n'.format(V - 1, num_features))
    vectors = SkipGram.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
    f.close()

    w2v = gensim.models.KeyedVectors.load_word2vec_format('./alignment_vec.txt', binary=False)
    print(w2v.most_similar(positive=['aa']))



def convert_base_pairs(x, y, max_document_length):
    #print('Max seq length:', max_document_length)
    #print('Input shape:', x.shape)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_proc = []
    y_proc = []
    for i in range(len(x)):
        #print('Alignment shape:', alignment.shape)
        proc = base_pairs_to_onehot(x[i][0], x[i][1], max_document_length)
        #proc = np_utils.to_categorical(proc, num_classes=120)
        #scaler = StandardScaler().fit(np.float64(proc))
        #proc = scaler.transform(np.float64(proc))
        if len(x_proc) == 0:
            x_proc = proc
            #x_proc = np.expand_dims(x_proc, axis=2)
            y_proc = y[i]
        else:
            #print('shape:', x_proc.shape, proc.shape)
            x_proc = np.vstack((x_proc, proc))
            y_proc = np.append(y_proc, y[i])
    #y_proc = np_utils.to_categorical(y_proc, num_classes=num_classes)
    #print('Labels shape', y_proc.shape)
    x_proc = np.reshape(x_proc, (y_proc.shape[0], -1))
    #print(x_proc)
    return x_proc, y_proc, max_document_length


def get_list_of_word2vec(x, w2v, max_length, n_samples):
    word_vec = []
    for alignment in x:
        vec = []
        word_list = alignment.split(' ')[:-1]
        for word in word_list:
            if len(vec) == 0:
                vec = w2v.word_vec(word)
                #vec = w2v.get_vector(word.replace('-',''))
            else:
                vec = np.vstack((vec, w2v.word_vec(word)))
        vec = np.reshape(vec, (len(word_list), -1))
        #print(vec)
        if len(word_vec) == 0:
            word_vec = vec
        else:
            #print('shapes', word_vec.shape, vec.shape)
            word_vec = np.dstack((word_vec, vec))
    #print(word_vec)
    #print(word_vec.shape)
    word_vec = np.reshape(word_vec, (n_samples, -1, num_features))
    return word_vec


def generate_word2vec_batch(x, y):
    while True:
        i = random.randint(0, len(x) - 1 - batch_size)
        j = random.randint(0, len(y) - 1 - batch_size)
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
        word2vec_list = get_list_of_word2vec(text, w2v, max_length, align_y.shape[0])
        align_y = np_utils.to_categorical(align_y)
        #print(word2vec_list, align_y)
        #print('shapes', word2vec_list.shape, align_y.shape)
        #align_x = np.reshape(word2vec_list, (align_x.shape[0], -1))
        yield word2vec_list, align_y


def generate_batch(x, y):
    while True:
        i = random.randint(0, len(x) - 1)
        j = random.randint(0, len(x) - 1)
        if (i + batch_size) < len(x) and (j + batch_size) < len(x):
            align_x, align_y, _ = (get_alignments(x, y, i, j, batch_size))
        elif (i + batch_size) >= len(x):
            print('End of training set, temp batch:', len(x[i:]))
            align_x, align_y, _ = (get_alignments(x, y, i, j, len(x[i:])))
        else:
            print('End of training set, temp batch:', len(x[j:]))
            align_x, align_y, _ = (get_alignments(x, y, i, j, len(x[j:])))
        #align_y = np.reshape(align_y, (len(align_y)//2, 2))
        align_x, align_y = convert_base_pairs(align_x, align_y)
        #align_x = np.expand_dims(align_x, axis=2)
        #align_x = np.reshape(align_x, (align_x.shape[0], align_x.shape[2], 2))
        #align_x = np.swapaxes(align_x, 0, 2)
        #print('Num Alignments', align_x.shape)
        #print('Labels:', align_y.shape)
        #print('Sample Alignment:', align_x[0], align_y[0])
        #print('Training on', i, j, '\n')
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

# split train data into train and validation
#sss = StratifiedShuffleSplit(test_size=0.2, random_state=23)
#for train_index, valid_index in sss.split(x_shuffle, y_shuffle):
#    x_train, x_valid = x_shuffle[train_index], y_shuffle[valid_index]
#    y_train, y_valid = x_shuffle[train_index], y_shuffle[valid_index]

x_train, x_valid, y_train, y_valid = train_test_split(x_shuffle,
                                                      y_shuffle,
                                                      stratify=y_shuffle,
                                                      test_size=0.5)

print('x shape:', x_train.shape)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(get_vocab('atcgx'))
V = len(tokenizer.word_index) + 1

#alignments2vec(x_train, y_train, V, tokenizer) #uncomment to train word2vec representation

model = Sequential()
#model.add(Embedding(25, num_features))
#model.add(Dropout(0.5))
model.add(Conv1D(filters=num_filters[0], kernel_size=50, input_shape=(None, num_features)))
model.add(Activation('relu'))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
model.add(Conv1D(filters=num_filters[1], kernel_size=5))
model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=num_features // batch_size, padding='valid'))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
# Shape is (batch_size, sentence_length)
#model.add(Conv2D(num_filters[0], kernel_size=(2,2), input_shape=(None, 2, 5)))
#model.add(Conv1D(nb_filter=num_filters[0], filter_length=50, input_shape=(None, 120)))
#model.add(Activation('relu'))
#model.add(Conv1D(nb_filter=num_filters[1], filter_length=5))
#model.add(Activation('relu'))
#model.add(Bidirectional(LSTM(hidden_size)))
model.add(AveragePooling1D(pool_size=int(num_features), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(1, 1)))
#model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
model.add(GlobalAveragePooling1D())
model.add(Dense(num_classes))
model.add(Activation('softmax'))
print(model.summary())

adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print('Training shapes:', x_train.shape, y_train.shape)
print('Valid shapes:', x_valid.shape, y_valid.shape)
'''
history = model.fit_generator(generate_batch(x_train, y_train),
                              steps_per_epoch=10,
                              epochs=10,#len(x_train)//batch_size//10,
                              validation_data=generate_batch(x_valid, y_valid),
                              validation_steps=10,
                              verbose=1)
'''
history = model.fit_generator(generate_word2vec_batch(x_train, y_train),
                              steps_per_epoch=10,
                              epochs=10,#len(x_train)//batch_size//10,
                              validation_data=generate_word2vec_batch(x_valid, y_valid),
                              validation_steps=10,
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
