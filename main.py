import numpy as np
import tensorflow as tf
import sys
import time
import logging
import os

import utils
import config

config = config.FLAGS

def _bi_rnn(inputs, seq_len, is_training=True, scope=None):
  '''
  return value:
    output:(output_fw, output_bw) [batch_size, max_time, hidden_size]
    state: (state_fw, state_bw) ([batch_size, hidden_size], ...) len() == num_layers
  '''
  def gru_cell():
    return tf.contrib.rnn.GRUCell(config.hidden_size)
  cell = gru_cell

  if is_training and config.dropout_rate < 1:
    def cell():
      return tf.contrib.rnn.DropoutWrapper(
            gru_cell(), output_keep_prob=config.dropout_rate)

  cell_fw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config.num_layers)] )
  cell_bw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config.num_layers)] )

  return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                          sequence_length=seq_len, dtype=tf.float32, scope=scope)

class Model(object):
  def __init__(self, embeddings, is_training=True):
    bz = config.batch_size
    ez = config.embedding_size
    hz = config.hidden_size * 2
    nl = config.num_labels

    # with tf.namespace('input'):
    doc = tf.placeholder(tf.int32, shape=[bz, None], name='document')
    doc_len = tf.placeholder(tf.int32, shape=[bz], name='doc_len')
    ques = tf.placeholder(tf.int32, shape=[bz, None], name='question')
    ques_len = tf.placeholder(tf.int32, shape=[bz], name='ques_len')
    label = tf.placeholder(tf.float32, shape=[bz, nl], name='label')
    ans = tf.placeholder(tf.int64, shape=[bz], name='answer')
    (self.doc, self.doc_len, self.ques, self.ques_len, self.label, self.ans) = \
                                  (doc, doc_len, ques, ques_len, label, ans)
    # with tf.namespace('embedding_lookup'):
    embeddings = tf.Variable(embeddings, dtype=tf.float32, name='embeddings')
    # embeddings = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='embeddings')
    doc_emb = tf.nn.embedding_lookup(embeddings, doc)
    ques_emb = tf.nn.embedding_lookup(embeddings, ques)


    # two bidirectional rnn, one for doc, one for question 
    output_d, state_d = _bi_rnn(doc_emb, doc_len, is_training, 'doc_rnn')
    output_q, state_q = _bi_rnn(ques_emb, ques_len, is_training, 'ques_rnn')
    
    d = tf.concat([output_d[0], output_d[1]], axis=2) # di = (fw_hi, bw_hi)
    # print(d.get_shape())#(32, 548, 256) (bz, len, hz)
    
    idx = tf.stack([tf.range(bz), ques_len-1], axis=1)
    q = tf.concat([tf.gather_nd(output_q[0], idx), output_q[1][:,0,:]], axis=1) # q = (fw_ht, bw_h1)
    # print(q.get_shape()) # (32, 256) (bz, hz)
    
    self.d = d

    # bilinear attention
    ws = tf.Variable(tf.random_uniform([hz, hz], minval=-0.01, maxval=0.01), name='ws')
    alpha = tf.matmul(q, ws) # (bz, hz)
    alpha = tf.matmul(d, tf.reshape(alpha,[bz, hz, 1]))
    alpha = tf.reshape(alpha, [bz, -1]) # (bz, len)
    alpha = tf.nn.softmax(alpha) # (bz, len)
    attention = tf.multiply(tf.reshape(alpha, [bz, -1, 1]), d) # (bz, len, hz)
    attention = tf.reduce_sum(attention, 1)# (bz, hz)

    # prediction 
    w = tf.Variable(tf.random_uniform([hz, nl], minval=-0.01, maxval=0.01), name='w') # [hz, nl]    
    logits = tf.matmul(attention, w) #[bz, nl] logits is unnormalized probabilities

    logits = logits * label # [bz, nl] [nl]
    pred = tf.arg_max(logits, -1) # [bz]
    
    acc = tf.reduce_sum(tf.cast(tf.equal(pred, ans), dtype=tf.int64))
    self.pred = pred
    self.acc = acc

    if not is_training:
      return
    
    # optimizer 
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(ans, nl))
    )
    optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      config.grad_clipping)
    capped_gvs = zip(grads, tvars)

    # tf.logging.set_verbosity(tf.logging.WARN)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    self.train_op = train_op
    self.loss = loss

def run_epoch(session, model, all_data, is_training=True, verbose=True):
  start_time = time.time()
  
  acc_count = 0
  total_steps = len(all_data)

  # np.random.shuffle(all_data)
  for step, minibatch in enumerate(all_data):
    (doc, doc_len, ques, ques_len, labled, ans) = minibatch

    # print(doc_len)
    # exit()


    feed_dict = {
      model.doc:doc, model.doc_len:doc_len, model.ques:ques,
      model.ques_len: ques_len, model.label: labled, model.ans: ans
    }
    
    if is_training:
      # d = session.run([model.d], feed_dict=feed_dict)
      # print('='*40)
      # print(d)
      # print('='*40)
      # exit()
      acc, loss = session.run([model.acc, model.loss], feed_dict=feed_dict)
      acc_count += acc
      if verbose:
        logging.info("  %.0f%% acc: %.2f%% loss: %.2f time: %.2f" %(
          step / total_steps * 100,
          acc_count / ((step+1) * config.batch_size) * 100,
          loss,
          time.time() - start_time
          ))
    else:
      acc, = session.run([model.acc], feed_dict=feed_dict)
      acc_count += acc
    
  return acc_count / (total_steps * config.batch_size)
    


def init():
  path = config.data_path
  config.embedding_file = os.path.join(path, config.embedding_file)
  config.train_file = os.path.join(path, config.train_file)
  config.dev_file = os.path.join(path, config.dev_file)
  config.test_file = os.path.join(path, config.test_file)
  
  dim = utils.get_dim(config.embedding_file)
  config.embedding_size = dim

  # Config log
  if config.log_file is None:
    logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  else:
    logging.basicConfig(filename=config.log_file,
                      filemode='w', level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  # Load data
  logging.info('-' * 50)
  logging.info('Load data files..')
  if config.debug:
    logging.info('*' * 10 + ' Train')
    train_examples = utils.load_data(config.train_file, 1000)
    logging.info('*' * 10 + ' Dev')
    dev_examples = utils.load_data(config.dev_file, 100)
  else:
    logging.info('*' * 10 + ' Train')
    train_examples = utils.load_data(config.train_file)
    logging.info('*' * 10 + ' Dev')
    dev_examples = utils.load_data(config.dev_file)

  config.num_train = len(train_examples[0])
  config.num_dev = len(dev_examples[0])

  # Build dictionary
  logging.info('-' * 50)
  logging.info('Build dictionary..')
  word_dict = utils.build_dict(train_examples[0] + train_examples[1])
  entity_markers = list(set( [w for w in word_dict.keys()
                            if w.startswith('@entity')] + train_examples[2] ))
  entity_markers = ['<unk_entity>'] + entity_markers
  entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
  logging.info('Entity markers: %d' % len(entity_dict))
  config.num_labels = len(entity_dict)

  logging.info('-' * 50)
  logging.info('Load embedding file..')
  embeddings = utils.gen_embeddings(word_dict, config.embedding_size, config.embedding_file)
  (config.vocab_size, config.embedding_size) = embeddings.shape

  # Log parameters
  flags = config.__dict__['__flags']
  flag_str = "\n"
  for k in flags:
    flag_str += "\t%s:\t%s\n" % (k, flags[k])
  logging.info(flag_str)

  # Vectorize test data
  logging.info('-' * 50)
  logging.info('Vectorize test data..')
  # d: document, q: question, a:answer
  # l: whether the entity label occurs in the document
  dev_d, dev_q, dev_l, dev_a = utils.vectorize(dev_examples, word_dict, entity_dict)
  assert len(dev_d) == config.num_dev
  all_dev = utils.gen_examples(dev_d, dev_q, dev_l, dev_a, config.batch_size)

  if config.test_only:
      return embeddings, all_dev, None

  # Vectorize training data
  logging.info('-' * 50)
  logging.info('Vectorize training data..')
  train_d, train_q, train_l, train_a = utils.vectorize(train_examples, word_dict, entity_dict)
  assert len(train_d) == config.num_train
  all_train = utils.gen_examples(train_d, train_q, train_l, train_a, config.batch_size)

  return embeddings, all_dev, all_train


def main(_):
  embeddings, all_dev, all_train = init()
  
  with tf.Graph().as_default():
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None, initializer=tf.truncated_normal_initializer(stddev=np.sqrt(0.1))): # tf.ones_initializer()
        m_train = Model(embeddings, is_training=True)
      # tf.summary.scalar("Training_Loss", m_train.loss)
      # tf.summary.scalar("Training_acc", m_train.acc)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", reuse=True):
        m_valid = Model(embeddings, is_training=False)
      # tf.summary.scalar("Valid_acc", m_valid.acc)
    
    sv = tf.train.Supervisor(logdir=config.save_path)
    with sv.managed_session() as session:

      if config.test_only:
        valid_acc = run_epoch(session, m_valid, all_dev, is_training=False)
        print("Valid acc: %.3f" % valid_acc)
      else:
        for epoch in range(config.num_epoches):
          # lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          # m.assign_lr(session, config.learning_rate * lr_decay)
          # np.random.shuffle(all_train)

          train_acc = run_epoch(session, m_train, all_train)
          logging.info("Epoch: %d Train acc: %.2f%%" % (epoch + 1, train_acc*100))
          valid_acc = run_epoch(session, m_valid, all_dev, is_training=False)
          logging.info("Epoch: %d Valid acc: %.2f%%" % (epoch + 1, valid_acc*100))
        # test_acc = run_epoch(session, m_test)
        if config.save_path:
          sv.saver.save(session, config.save_path, global_step=sv.global_step)

  
if __name__ == '__main__':
  tf.app.run()