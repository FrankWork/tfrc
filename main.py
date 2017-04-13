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
  def __init__(self, embeddings, is_training):
    bz = config.batch_size
    ez = config.embedding_size

    # with tf.namespace('input'):
    doc = tf.placeholder(tf.int32, shape=[bz, None], name='document')
    doc_len = tf.placeholder(tf.int32, shape=[bz], name='doc_len')
    ques = tf.placeholder(tf.int32, shape=[bz, None], name='question')
    ques_len = tf.placeholder(tf.int32, shape=[bz], name='ques_len')
    label = tf.placeholder(tf.float32, shape=[bz, config.num_labels], name='label')
    ans = tf.placeholder(tf.int32, shape=[bz], name='answer')
    (self.doc, self.doc_len, self.ques, self.ques_len, self.label, self.ans) = \
                                  (doc, doc_len, ques, ques_len, label, ans)
    # with tf.namespace('embedding_lookup'):
    embeddings = tf.Variable(embeddings, dtype=tf.float32, trainable=False,name='embeddings')
    # embeddings = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='embeddings')
    doc_emb = tf.nn.embedding_lookup(embeddings, doc)
    ques_emb = tf.nn.embedding_lookup(embeddings, ques)


    # two bidirectional rnn, one for doc, one for question 
    # inputs = tf.unpack(x, n_steps, 1)
    output_d, state_d = _bi_rnn(doc_emb, doc_len, is_training, 'doc_rnn')
    output_q, state_q = _bi_rnn(ques_emb, ques_len, is_training, 'ques_rnn')
    

    self.output_d = output_d
    self.output_q = output_q
    # bilinear attention
    # alpha = softmax(output_q.transpose * W * output_d)
    # o = tf.reduce_sum(alpha * output_d)


    # optimizer 
  
  

def train(embeddings,train_examples, word_dict, entity_dict):
  # Training
  logging.info('-' * 50)
  logging.info('Start training..')
  train_d, train_q, train_l, train_a = utils.vectorize(train_examples, word_dict, entity_dict)
  assert len(train_d) == config.num_train
  all_train = utils.gen_examples(train_d, train_q, train_l, train_a, config.batch_size)  
  start_time = time.time()
  n_updates = 0

  model = Model(embeddings, True)

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(config.num_epoches):
      # np.random.shuffle(all_train)
      for idx, minibatch in enumerate(all_train):
        (doc, doc_len, ques, ques_len, labled, ans) = minibatch
        # (doc, doc_len, ques, ques_len, label, ans) = model.inputs
        # for item in minibatch:
        #   print(type(item))
        # exit()
        feed_dict = {model.doc:doc, model.doc_len:doc_len, model.ques:ques, \
                  model.ques_len: ques_len, model.label: labled, model.ans: ans}
        o1, o2 = session.run([model.output_d, model.output_q], feed_dict=feed_dict)
        print('*' * 40)
        print(o1)
        print('*' * 40)
        print(o2)
        # print('*' * 40)
        # print(ques)
        # print('*' * 40)
        # print(labled[0])
        # assert len(labled[0]) == config.num_labels
        # print('*' * 40)
        # print(ans)
      
        exit()

        # TODO: training

  

def main(_):
  path = config.data_path
  config.embedding_file = os.path.join(path, config.embedding_file)
  config.train_file = os.path.join(path, config.train_file)
  config.dev_file = os.path.join(path, config.dev_file)
  config.test_file = os.path.join(path, config.test_file)

  # print(config.embedding_file)
  # exit()
  dim = utils.get_dim(config.embedding_file)
  config.embedding_size = dim

  if config.log_file is None:
    logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  else:
    logging.basicConfig(filename=config.log_file,
                      filemode='w', level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  
  logging.info('-' * 50)
  logging.info('Load data files..')
  if config.debug:
    logging.info('*' * 10 + ' Train')
    train_examples = utils.load_data(config.train_file, 100)
    logging.info('*' * 10 + ' Dev')
    dev_examples = utils.load_data(config.dev_file, 100)
  else:
    logging.info('*' * 10 + ' Train')
    train_examples = utils.load_data(config.train_file)
    logging.info('*' * 10 + ' Dev')
    dev_examples = utils.load_data(config.dev_file)

  config.num_train = len(train_examples[0])
  config.num_dev = len(dev_examples[0])

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

  # log parameters
  flags = config.__dict__['__flags']
  flag_str = "\n"
  for k in flags:
    flag_str += "\t%s:\t%s\n" % (k, flags[k])
  logging.info(flag_str)

  # logging.info('Compile functions..')
  # train_fn, test_fn, params = build_fn(config, embeddings)
  # logging.info('Done.')

  logging.info('-' * 50)
  logging.info('Intial test..')
  # d: document, q: question, a:answer
  # l: whether the entity label occurs in the document
  dev_d, dev_q, dev_l, dev_a = utils.vectorize(dev_examples, word_dict, entity_dict)
  assert len(dev_d) == config.num_dev
  all_dev = utils.gen_examples(dev_d, dev_q, dev_l, dev_a, config.batch_size)
  # dev_acc = eval_acc()# TODO
  # logging.info('Dev accuracy: %.2f %%' % dev_acc)
  # best_acc = dev_acc

  if config.test_only:
      return
  
  train(embeddings, train_examples, word_dict, entity_dict)
  
if __name__ == '__main__':
  tf.app.run()