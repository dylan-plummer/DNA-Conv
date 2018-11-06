import numpy as np
import pandas as pd
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv1D, AveragePooling1D, MaxPooling1D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt
import data_helpers as dhrt


# Network Parameters
learning_rate = 0.001
num_classes = 2
num_features = 372
batch_size = 64
nb_epoch = 64

# load data
x_rt, y_rt = dhrt.load_data_and_labels('h3k4me3.pos', 'h3k4me3.neg')
lens = [len(x.split(" ")) for x in x_rt]
max_document_length = max(lens)

if max_document_length%2 != 0:
    max_document_length=max_document_length+1

print( "Max Document Length = ", max_document_length)
print( "Number of Samples =", len(y_rt))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_rt_proc = np.array(list(vocab_processor.fit_transform(x_rt)))
l_x_rt = len(y_rt)
vocab_size = len(vocab_processor.vocabulary_)

print( "Vocab Size: ", vocab_size)
num_classes = 2
num_filters = [16, 4]

region_size = 51 #can be considered as filter size but not really

#this value has to be selected based on max_document_length and region_size
#here I ensured that max_docu length is even and region size in odd
#so division by 2 is possible
num_pooled = (max_document_length-region_size+1)/2

print( "Pool Size: ", num_pooled)


#np.random.seed(10) #for 'reproducible research' B-)
#^comment this line out to evaluate the model more thoroughly
#in case this line is commented out, standard cross validation is assumed
#so, a separate test dataset is necessary -
#in order to ensure the model(s) have been trained properly
#this code does not save the model in a file
#somene has to implement
#^NOTES or TO DO

shuffled_rt = np.random.permutation(range(l_x_rt))
x_rt_shuffled = x_rt_proc[shuffled_rt]
y_rt_shuffled = y_rt[shuffled_rt]

# standardize train features
scaler = StandardScaler().fit(x_rt_shuffled)
scaled_train = scaler.transform(x_rt_shuffled)

# split train data into train and validation
sss = StratifiedShuffleSplit(test_size=0.2, random_state=23)
for train_index, valid_index in sss.split(scaled_train, y_rt_shuffled):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = y_rt_shuffled[train_index], y_rt_shuffled[valid_index]

# reshape train data
print('X_train',X_train.shape)
X_train_r = np.expand_dims(X_train, axis=2)
X_valid_r = np.expand_dims(X_valid, axis=2)
X_train_r = np.tile(X_train_r, (1, 1, batch_size))
print(X_train_r.shape)
X_train_r = np.reshape(X_train_r, (X_train_r.shape[0], batch_size, X_train_r.shape[1]))
X_valid_r = np.tile(X_valid_r, (1, 1, batch_size))
X_valid_r = np.reshape(X_valid_r, (X_valid_r.shape[0], batch_size, X_valid_r.shape[1]))


filter_length = vocab_size*region_size
sentence_length = x_rt_proc.shape[1]*vocab_size
cnn_filter_shape = [filter_length, 1, 1, num_filters[0]]
pool_stride = [1,int((x_rt_proc.shape[1]-region_size+1)/num_pooled), 1, 1]
print('SHAPE?', X_train_r.shape)
model = Sequential()
# Shape is (batch_size, sentence_length)
model.add(Conv1D(nb_filter=num_filters[0], filter_length=batch_size//8, input_shape=(batch_size, X_train_r.shape[2])))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=num_features//batch_size, padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Conv1D(nb_filter=num_filters[1], filter_length=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=int(num_pooled), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
#model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
#model.add(Dense(512, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))
print(model.summary())

adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train_r, y_train, nb_epoch=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=batch_size)
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