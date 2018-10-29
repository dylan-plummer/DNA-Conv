import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import data_helpers as dhrt

# Training Parameters
learning_rate = 0.001
num_steps = 2000
num_epochs = 100
batch_size = 64

#load data
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

# Network Parameters
num_classes = 2
num_filters = [16, 8]
region_size = 51 #can be considered as filter size but not really
#this value has to be selected based on max_document_length and region_size
#here I ensured that max_docu length is even and region size in odd
#so division by 2 is possible
num_pooled = (max_document_length-region_size+1)/2
dropout = 0.25


def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    x = tf.cast(tf.reshape(x, [-1, 500, 64]), dtype=tf.float32)
    with tf.variable_scope('ConvNet', reuse=reuse):
        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv1d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling1d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv1d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling1d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    labels = tf.cast(tf.expand_dims(labels, axis=0), tf.float32)
    #features = tf.cast(tf.expand_dims(features, axis=0), tf.float32)
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    print(labels)
    print(logits_train)
    print(tf.reduce_max(labels))
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


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
y_rt_shuffled = np.argmax(y_rt_shuffled, axis=1)
print(y_rt_shuffled)

rt_split_size = 3000
x_rt_train = x_rt_shuffled[:-rt_split_size]
x_rt_val = x_rt_shuffled[-rt_split_size:]
y_rt_train = y_rt_shuffled[:-rt_split_size]
y_rt_val = y_rt_shuffled[-rt_split_size:]

rt_train_length = len(y_rt_train)
rt_val_length = len(y_rt_val)

print(y_rt_train)


# Build the Estimator
model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x=x_rt_train,
    y=y_rt_train,
    batch_size=batch_size,
    num_epochs=None,
    shuffle=True
)
# Train the model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x=x_rt_val,
    y=y_rt_val,
    batch_size=batch_size,
    shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])

