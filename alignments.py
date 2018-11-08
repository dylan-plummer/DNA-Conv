import data_helpers as dhrt
from tensorflow.contrib import learn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, LSTM, Bidirectional, BatchNormalization
from keras.optimizers import SGD, Adam
from Bio import pairwise2


# Network Parameters
learning_rate = 0.001
num_classes = 2
num_features = 372
batch_size = 4
nb_epoch = 4
hidden_size = 100
num_sequences = 10
num_classes = 2
num_filters = [16, 4]


def replace_spaces(x):
    return x.replace(' ', '')


def get_alignments(x, y, i, j, batch_size):
    '''
    Aligns every pair of sequences to prepare input to CNN
    :param x: a set of sequences
    :param y: a set of labels
    :return: a set of pairwise alignments of the sets in x (cartesian product) x
    '''
    a = pairwise2.align.globalxx(x[i],x[j])[0:2]
    align_x = np.array(list(a)).T[0:2]
    if np.array_equal(y[i], y[j]):
        align_y = [[1, 0],[1,0]]
    else:
        align_y = [[0, 1],[0,1]]
    for i in range(i + 1, i + batch_size):
        for j in range(j + 1, j + batch_size):
            if i < j:
                a = pairwise2.align.globalxx(x[i],x[j])[0:2]
                align_x = np.append(align_x, np.array(list(a)).T[0:2], axis=1)
                if np.array_equal(y[i], y[j]):
                    align_y = np.append(align_y, [[1, 0],[1, 0]], axis=1)
                else:
                    align_y = np.append(align_y, [[1, 0],[1, 0]], axis=1)
    return align_x, align_y


def convert_base_pairs(x):
    lens = [len(seq) for alignment in x for seq in alignment]
    max_document_length = max(lens)
    if max_document_length % 2 != 0:
        max_document_length = max_document_length + 1
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_proc = []
    for i in range(len(x)):
        alignment = []
        for j in range(len(x[i])):
            proc = np.array(list(vocab_processor.fit_transform(x[i][j])))
            scaler = StandardScaler().fit(proc)
            proc = scaler.transform(proc)
            if len(alignment)==0:
                alignment = proc
            else:
                alignment = np.vstack((alignment, proc))
        if len(x_proc) == 0:
            x_proc = alignment
        else:
            x_proc = np.vstack((x_proc, alignment))
        print(x_proc)
        print(x_proc.shape)
    return np.reshape(x_proc, (len(x_proc)//2//max_document_length, 2, max_document_length))


def generate_batch(x, y):
    for i in range(0, len(x), batch_size):
        for j in range(0, len(y), batch_size):
            align_x, align_y = (get_alignments(x, y, i, j, batch_size))
            #align_y = np.reshape(align_y, (len(align_y)//2, 2))
            align_x = convert_base_pairs(align_x)
            align_x = np.swapaxes(align_x, 1, 2)
            print(align_x)
            print(align_x)
            print(align_y)
            print('Num Alignments', align_x.shape)
            print('Labels:', align_y.shape)
            print('Sample Alignment:', align_x[0], align_y[0])
            yield align_x, align_y


# load data
x_rt, y_rt = dhrt.load_data_and_labels('h3.pos', 'h3.neg')

x_rt = np.array([replace_spaces(seq) for seq in x_rt])
print('X:', x_rt)
y_rt = np.array(list(y_rt))
print(pairwise2.align.globalxx('aaattcgctgc','aaatctcgcgat'))
shuffled_rt = np.random.permutation(range(len(x_rt)))
x_shuffle = x_rt[shuffled_rt]
y_shuffle = y_rt[shuffled_rt]

# split train data into train and validation
sss = StratifiedShuffleSplit(test_size=0.2, random_state=23)
for train_index, valid_index in sss.split(x_shuffle, y_shuffle):
    x_train, x_valid = x_shuffle[train_index], y_shuffle[valid_index]
    y_train, y_valid = x_shuffle[train_index], y_shuffle[valid_index]

print('x shape:', x_train.shape)

model = Sequential()
# Shape is (batch_size, sentence_length)
model.add(Conv1D(nb_filter=num_filters[0], filter_length=2, input_shape=(None, 2)))
model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=num_features//batch_size, padding='valid'))
#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(Conv1D(nb_filter=num_filters[1], filter_length=1))
#model.add(Activation('relu'))
model.add(AveragePooling1D(pool_size=int(num_features), padding='same'))
model.add(Activation('relu'))
model.add(Bidirectional(LSTM(hidden_size)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
#model.add(Flatten())
#model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))
print(model.summary())

adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit_generator(generate_batch(x_train, y_train),
                    steps_per_epoch=50,
                    epochs=nb_epoch,
                    validation_data=generate_batch(x_valid, y_valid),
                    validation_steps=50,
                    verbose=1)