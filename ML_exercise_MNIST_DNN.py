import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope

from tensorflow.examples.tutorials.mnist import input_data

from tqdm import *

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# params
learning_rate = 0.015
training_epochs = 15
batch_size = 100
keep_prob = 0.7

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
train_mode = tf.placeholder(tf.bool, name='train_mode')

hidden_output_size = 1024
final_output_size = 10

variance_init = tf.contrib.layers.variance_scaling_initializer()
bn_params = {
    'is_training': train_mode,
    'decay': 0.9,
    'updates_collections': None
}

with arg_scope([fully_connected],
               activation_fn=tf.sigmoid,
               weights_initializer=variance_init,
               biases_initializer=None,
               normalizer_fn=batch_norm,
               normalizer_params=bn_params):
    hidden_layer1 = fully_connected(X, hidden_output_size, scope="h1")
    h1_drop = dropout(hidden_layer1, keep_prob, is_training=train_mode)
    hidden_layer2 = fully_connected(h1_drop, hidden_output_size, scope="h2")
    h2_drop = dropout(hidden_layer2, keep_prob, is_training=train_mode)
    hidden_layer3 = fully_connected(h2_drop, hidden_output_size, scope="h3")
    h3_drop = dropout(hidden_layer3, keep_prob, is_training=train_mode)
    hidden_layer4 = fully_connected(h3_drop, hidden_output_size, scope="h4")
    h4_drop = dropout(hidden_layer4, keep_prob, is_training=train_mode)
    hypothesis = fully_connected(
        h4_drop, final_output_size, activation_fn=None, scope="hypothesis")
# cost/lost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# init
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
print("epoch = {}, layer = 5, width = {}, keep_prob = {}, learning_rate = {}".format(
    training_epochs, hidden_output_size, keep_prob, learning_rate))
for epoch in tqdm(range(training_epochs)):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in tqdm(range(total_batch)):
        xs, ys = mnist.train.next_batch(batch_size)
        feed_dict_train = {X: xs, Y: ys, train_mode: True}
        feed_dict_cost = {X: xs, Y: ys, train_mode: False}
        opt = sess.run(optimizer, feed_dict=feed_dict_train)
        c = sess.run(cost, feed_dict=feed_dict_cost)
        avg_cost += c / total_batch

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost))

print('Learning Finished!')

# test
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy', sess.run(accuracy, feed_dict={
    X: mnist.test.images, Y: mnist.test.labels, train_mode: False
}))

'''
epoch = 15, layer = 5, width = 512, keep_prob = 0.7
[Epoch:    1] cost = 0.377187196
[Epoch:    2] cost = 0.320057436
[Epoch:    3] cost = 0.311145618
[Epoch:    4] cost = 0.304886997
[Epoch:    5] cost = 0.301448307
[Epoch:    6] cost = 0.297418495
[Epoch:    7] cost = 0.296424182
[Epoch:    8] cost = 0.295339055
[Epoch:    9] cost = 0.293299217
[Epoch:   10] cost = 0.292921558
[Epoch:   11] cost = 0.291565194
[Epoch:   12] cost = 0.289973097
[Epoch:   13] cost = 0.29060266
[Epoch:   14] cost = 0.291325585
[Epoch:   15] cost = 0.289528506
Learning Finished!
Accuracy 0.9829
'''

'''
epoch = 15, layer = 5, width = 1024, keep_prob = 0.7
[Epoch:    1] cost = 0.376792876
[Epoch:    2] cost = 0.317857504
[Epoch:    3] cost = 0.308155891
[Epoch:    4] cost = 0.302109988
[Epoch:    5] cost = 0.299020546
[Epoch:    6] cost = 0.295795071
[Epoch:    7] cost = 0.294304553
[Epoch:    8] cost = 0.293089727
[Epoch:    9] cost = 0.291036452
[Epoch:   10] cost = 0.291049045
[Epoch:   11] cost = 0.290474638
[Epoch:   12] cost = 0.288450405
[Epoch:   13] cost = 0.289603117
[Epoch:   14] cost = 0.289308299
[Epoch:   15] cost = 0.289242009
Learning Finished!
Accuracy 0.9853
'''

'''
epoch = 15, layer = 5, width = 1024, keep_prob = 0.6
[Epoch:    1] cost = 0.373880765
[Epoch:    2] cost = 0.316476893
[Epoch:    3] cost = 0.308174907
[Epoch:    4] cost = 0.302932075
[Epoch:    5] cost = 0.298467432
[Epoch:    6] cost = 0.295690488
[Epoch:    7] cost = 0.294116734
[Epoch:    8] cost = 0.292477789
[Epoch:    9] cost = 0.290795798
[Epoch:   10] cost = 0.29139414
[Epoch:   11] cost = 0.289932299
[Epoch:   12] cost = 0.28952417
[Epoch:   13] cost = 0.289305165
[Epoch:   14] cost = 0.288189703
[Epoch:   15] cost = 0.288602556
Learning Finished!
Accuracy 0.9849
'''

'''
epoch = 15, layer = 5, width = 1024, keep_prob = 0.7, learning_rate = 0.015
[Epoch:    1] cost = 0.420254938
[Epoch:    2] cost = 0.318560145
[Epoch:    3] cost = 0.309397926
[Epoch:    4] cost = 0.302606855
[Epoch:    5] cost = 0.299062064
[Epoch:    6] cost = 0.295539058
[Epoch:    7] cost = 0.293949145
[Epoch:    8] cost = 0.292840702
[Epoch:    9] cost = 0.291011021
[Epoch:   10] cost = 0.290140437
[Epoch:   11] cost = 0.290367571
[Epoch:   12] cost = 0.288814293
[Epoch:   13] cost = 0.289330706
[Epoch:   14] cost = 0.289183544
[Epoch:   15] cost = 0.288788885
Learning Finished!
Accuracy 0.9846
'''

'''
epoch = 15, layer = 5, width = 1024, keep_prob = 0.7, learning_rate = 0.015, activation_fn = tf.sigmoid
[Epoch:    1] cost = 0.610473796
[Epoch:    2] cost = 0.456135646
[Epoch:    3] cost = 0.392004524
[Epoch:    4] cost = 0.361062161
[Epoch:    5] cost = 0.343882866
[Epoch:    6] cost = 0.3319505
[Epoch:    7] cost = 0.3210544
[Epoch:    8] cost = 0.312056473
[Epoch:    9] cost = 0.303079894
[Epoch:   10] cost = 0.299181798
[Epoch:   11] cost = 0.292175158
[Epoch:   12] cost = 0.289107808
[Epoch:   13] cost = 0.286365732
[Epoch:   14] cost = 0.28207678
[Epoch:   15] cost = 0.281006887
Learning Finished!
Accuracy 0.9836
'''
