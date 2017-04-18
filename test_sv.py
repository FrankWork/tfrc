import tensorflow as tf
import numpy as np



def _bi_rnn():
  w = tf.get_variable(name='w', shape=[2,2], dtype=tf.int32, initializer=tf.ones_initializer())
  return w

class Model(object):
  def __init__(self):
    with tf.variable_scope("_bi_rnn", reuse=None):
      w = _bi_rnn()
    # w = tf.Variable(tf.ones([2,2], dtype=tf.int32))
    self.g = tf.Variable(0, name='global_step')
    with tf.control_dependencies([tf.assign_add(self.g, 1)]):
      w = tf.assign_add(w, tf.ones([2, 2], dtype=tf.int32))    
    self.w = w

def run_epoch(session, model):
  for i in range(5):
    print(session.run([model.w]))

def main():
  with tf.Graph().as_default():
      with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None):
          model = Model()
      sv = tf.train.Supervisor(logdir='model/', summary_op=None, save_model_secs=0, global_step=model.g)
      with sv.managed_session() as session:
        for epoch in range(5):
          # print('epoch %d step %d' % (epoch, sv.global_step))
          run_epoch(session, model)
        sv.saver.save(session, 'model/', global_step=sv.global_step)

main()