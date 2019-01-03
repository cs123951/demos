import os.path

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST')

input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

x = tf.placeholder("float", shape=[None, input_dim])

W_encoder_input_hidden = weight_variable([input_dim, hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])


saver = tf.train.Saver()
n_steps = int(1e6)
batch_size = 100

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('experiment',
                                          graph=sess.graph)
    if os.path.isfile("save/model.ckpt"):
        print("Restoring saved parameters")
        saver.restore(sess, "save/model.ckpt")
    else:
        print("Initializing parameters")
        sess.run(tf.global_variables_initializer())
        for step in range(1, n_steps):
            batch = mnist.train.next_batch(batch_size)
            feed_dict = {x: batch[0]}
            _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

            if step % 50 == 0:
                save_path = saver.save(sess, "save/model.ckpt")
                print("Step {0} | Loss: {1}".format(step, cur_loss))