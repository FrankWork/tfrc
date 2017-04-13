import tensorflow as tf

hidden_size = 1
num_layers = 1
dropout_rate = 0.2
batch_size = 2
max_time = 3
embedding_size = 1


def _bi_rnn(inputs, seq_len, is_training=True):
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

  return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=seq_len, dtype=tf.float32)


# [batch_size, max_time, embedding_size]
inputs = tf.convert_to_tensor([

])

tf.ones([batch_size, max_time, embedding_size])

seq_len = tf.ones(batch_size, dtype=tf.int32) * max_time

o, s = _bi_rnn(inputs, seq_len)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    o, s= session.run([o, s])
    print(o[0])
    print('*' * 40)
    print(o[1])
    print('*' * 40)
    print(s[0])
    print('*' * 40)
    print(s[1])
# [[[-0.        ]
#   [-0.        ]
#   [-0.        ]]

#  [[-0.        ]
#   [-0.95163113]
#   [-0.        ]]]
# ****************************************
# [[[ 0.        ]
#   [ 1.55676079]
#   [ 0.        ]]

#  [[ 0.        ]
#   [ 0.        ]
#   [ 0.        ]]]
# ****************************************
# (array([[-0.19355205],
#        [-0.19355205]], dtype=float32),)
# ****************************************
# (array([[ 0.3127957],
#        [ 0.3127957]], dtype=float32),)
