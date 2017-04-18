import tensorflow as tf
import numpy as np



def _bi_rnn():
  # w = tf.get_variable(name='w', shape=[2,2], dtype=tf.int32, initializer=tf.ones_initializer())
  w = tf.get_variable(name='w', dtype=tf.int32, initializer=[[1, 1], [1, 1]])
  
  return w

class Model(object):
  def __init__(self):
    with tf.variable_scope("_bi_rnn", reuse=None):
      w = _bi_rnn()
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
        with tf.variable_scope("model", reuse=None):
          train = Model()
      with tf.name_scope('Test'):
        with tf.variable_scope("model", reuse=True):
          test = Model()
      sv = tf.train.Supervisor(logdir='model/', summary_op=None,  global_step=train.g)# save_model_secs=600,
      with sv.managed_session() as session:
        for epoch in range(5):
          # print('epoch %d step %d' % (epoch, sv.global_step))
          run_epoch(session, train)
        print('test:')
        print(session.run([test.w]))
        sv.saver.save(session, 'model/', global_step=sv.global_step)

main()