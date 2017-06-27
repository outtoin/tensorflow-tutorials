import tensorflow as tf
import numpy as np
from tqdm import tqdm_gui as tqdm
tf.set_random_seed(777)


idx2char = ['h', 'i', 'e', 'l', 'o']


x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [[1, 0, 2, 3, 3, 4]]

num_classes = 5
input_dim = 5
hidden_size = 5
batch_size = 1
sequence_length = 6
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
                logits=outputs, targets=Y, weights=weights)
# X_for_fc = tf.reshape(outputs, [-1, hidden_size])

loss = tf.reduce_mean(sequence_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(2000)):
        l, _  = sess.run([loss, optimizer], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})

        if i % 200 == 0:
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tPrediction String: ", ''.join(result_str))
