import data_helpers as dhrt
from tensorflow.contrib import learn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from Bio import pairwise2


# Network Parameters
learning_rate = 0.001
num_classes = 2
num_features = 372
batch_size = 128
nb_epoch = 100
hidden_size = 100
num_sequences = 10

# load data
x_rt, y_rt = dhrt.load_data_and_labels('h3.pos', 'h3.neg')


def replace_spaces(x):
    return x.replace(' ', '')


x_rt = np.array([replace_spaces(seq) for seq in x_rt])
print('X:', x_rt)
y_rt = np.array(list(y_rt))
print(pairwise2.align.globalxx('aaattcgctgc','aaatctcgcgat'))


def get_alignments(x, y):
    '''
    Aligns every pair of sequences to prepare input to CNN
    :param x: a set of sequences
    :param y: a set of labels
    :return: a set of pairwise alignments of the sets in x (cartesian product) x
    '''
    align_x = []
    align_y = []
    for i in range(len(x)):
        for j in range(len(x)):
            if i < j:
                a = pairwise2.align.globalxx(x[i],x[j])[0:2]
                try:
                    align_x = np.append(align_x, np.array(list(a)).T[0:2])
                    if np.array_equal(y[i], y[j]):
                        align_y = np.append(align_y, [1, 0])
                        align_y = np.append(align_y, [1, 0])
                    else:
                        align_y = np.append(align_y, [0, 1])
                        align_y = np.append(align_y, [0, 1])
                except Exception as e:
                    print('No alignments')
                    print(align_x.shape, 'vs.', np.array(list(a)).T[0:2].shape)
    return align_x, align_y


align_x, align_y = (get_alignments(x_rt[0:num_sequences], y_rt[0:num_sequences]))
print(align_x)
print(align_y)
print('Num Alignments', align_x.shape)
lens = [len(x) for x in align_x]
max_document_length = max(lens)
if max_document_length%2 != 0:
    max_document_length=max_document_length+1

print( "Max Document Length = ", max_document_length)
print( "Number of Samples =", len(y_rt))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_rt_proc = np.array(list(vocab_processor.fit_transform(align_x)))
l_x_rt = len(align_x)
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
print(align_y)
x_rt_shuffled = align_x[shuffled_rt]
y_rt_shuffled = align_y[shuffled_rt]

# standardize train features
scaler = StandardScaler().fit(x_rt_shuffled)
scaled_train = scaler.transform(x_rt_shuffled)

# split train data into train and validation
sss = StratifiedShuffleSplit(test_size=0.2, random_state=23)
for train_index, valid_index in sss.split(scaled_train, y_rt_shuffled):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = y_rt_shuffled[train_index], y_rt_shuffled[valid_index]

def generate_sample(x1, x2):
    return x1