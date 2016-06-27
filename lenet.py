import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels): #
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 6
c5_layer = 120
f6_layer = 84

graph = tf.Graph()

with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    c1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    c1_biases = tf.Variable(tf.zeros([depth]))

    c3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, 12], stddev=0.1))
    c3_biases = tf.Variable(tf.constant(1.0, shape=[12]))

    c5_weights = tf.Variable(tf.truncated_normal(
      [patch_size * patch_size * 1 * 12, c5_layer], stddev=0.1))
    c5_biases = tf.Variable(tf.constant(1.0, shape=[c5_layer]))

    f6_weights = tf.Variable(tf.truncated_normal(
      [c5_layer, f6_layer], stddev=0.1))
    f6_biases = tf.Variable(tf.constant(1.0, shape=[f6_layer]))

    out_weights = tf.Variable(tf.truncated_normal([f6_layer, num_labels],stddev=0.1))
    out_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

# Model.
    def model(data):
        conv = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + c1_biases)
        ##pooling1
        pooling_layer = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

        conv = tf.nn.conv2d(pooling_layer, c3_weights, [1,1,1,1],padding='VALID')
        hidden = tf.nn.relu(conv + c3_biases)
        ##pooling2
        pooling_layer = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

        shape = pooling_layer.get_shape().as_list()
        reshape = tf.reshape(pooling_layer, [shape[0], shape[1] * shape[2] * shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, c5_weights) + c5_biases)

#        c5_dropout = tf.nn.dropout(hidden, 0.5)

        hidden = tf.nn.relu(tf.matmul(hidden, f6_weights) + f6_biases)

        return tf.matmul(hidden, out_weights) + out_biases
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    regularizers = (tf.nn.l2_loss(c5_weights) + tf.nn.l2_loss(c5_biases)+
                    tf.nn.l2_loss(f6_weights) + tf.nn.l2_loss(f6_biases))

    loss += 5e-4 * regularizers

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 100 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))