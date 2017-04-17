import numpy as np
import tensorflow as tf


dropout_rate = 1 # keep_rate
hidden_size = 3
num_layers = 1

def _bi_rnn(inputs, seq_len, is_training=True, scope=None):
  '''
  return value:
    output:(output_fw, output_bw) [batch_size, max_time, hidden_size]
    state: (state_fw, state_bw) ([batch_size, hidden_size], ...) len() == num_layers
  '''
  def gru_cell():
    return tf.contrib.rnn.GRUCell(hidden_size)
  cell = gru_cell

  if is_training and dropout_rate < 1:
    def cell():
      return tf.contrib.rnn.DropoutWrapper(
            gru_cell(), output_keep_prob=dropout_rate)

  cell_fw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(num_layers)] )
  cell_bw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(num_layers)] )

  return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                          sequence_length=seq_len, dtype=tf.float32, scope=scope)

doc = np.ones([2, 3, 1], dtype=np.float32)
len = np.array([3, 3])

with tf.variable_scope('bi_rnn', initializer=tf.ones_initializer()) as vs:
    output, state = _bi_rnn(doc, len)

# vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
# for v in vars:
#     print(v.op.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o = sess.run([output])
    print(o)