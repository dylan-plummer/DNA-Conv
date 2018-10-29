import numpy as np
import pandas as pd
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import data_helpers as dhrt


# Network Parameters
num_classes = 2
num_features = 372
batch_size = 16

# load data
x_rt, y_rt = dhrt.load_data_and_labels('h3.pos','h3.neg')
lens = [len(x.split(" ")) for x in x_rt];
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
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, valid_index in sss.split(scaled_train, y_rt_shuffled):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = y_rt_shuffled[train_index], y_rt_shuffled[valid_index]

# reshape train data
X_train_r = np.zeros((len(X_train), num_features, 3))

# reshape validation data
X_valid_r = np.zeros((len(X_valid), num_features, 3))

# Keras model with one Convolution1D layer
# unfortunately more number of covnolutional layers, filters and filters lenght
# don't give better accuracy
model = Sequential()
model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(num_features, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

y_rt_train = np_utils.to_categorical(y_train, num_classes)
y_rt_val = np_utils.to_categorical(y_train, num_classes)

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

nb_epoch = 15
model.fit(X_train_r, y_train, nb_epoch=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=batch_size)