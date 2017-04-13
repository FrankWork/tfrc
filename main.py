import numpy as np
import tensorflow as tf
import sys
import time
import logging

import utils
import config

config = config.FLAGS

def _bi_rnn(inputs, seq_len):
  def gru_cell():
    return tf.contrib.rnn.GRUCell(config.hidden_size)
  cell = gru_cell

  if is_training and config.dropout_rate < 1:
    def cell():
      return tf.contrib.rnn.DropoutWrapper(
            gru_cell(), output_keep_prob=config.dropout_rate)

  cell_fw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config.num_layers)] )
  cell_bw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config.num_layers)] )

  # return value: output, state
  # output shape:(output_fw, output_bw) [batch_size, max_time, hidden_size]
  # state shape: (state_fw, state_bw) ([batch_size, hidden_size], ...) len() == num_layers
  return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=seq_len, dtype=tf.float32)

def build_model(embeddings, is_training):
  bz = config.batch_size
  ez = config.embedding_size

  # with tf.namespace('input'):
  doc = tf.placeholder(tf.int32, shape=[bz, None], name='document')
  doc_mask = tf.placeholder(tf.float32, shape=[bz, None], name='doc_mask')
  ques = tf.placeholder(tf.int32, shape=[bz, None], name='question')
  ques_mask = tf.placeholder(tf.float32, shape=[bz, None], name='ques_mask')
  label = tf.placeholder(tf.float32, shape=[bz], name='label')
  ans = tf.placeholder(tf.int32, shape=[bz], name='answer')

  # with tf.namespace('embedding_lookup'):
  doc_emb = tf.nn.embedding_lookup(embeddings, doc)
  ques_emb = tf.nn.embedding_lookup(embeddings, ques)


  # two bidirectional rnn, one for doc, one for question 
  # inputs = tf.unpack(x, n_steps, 1)
  output_d, state_d = _bi_rnn(doc)
  output_q, state_q = _bi_rnn(ques)
  


  # bilinear attention
  # alpha = softmax(output_q.transpose * W * output_d)
  # o = tf.reduce_sum(alpha * output_d)


  # optimizer 
  

  

def main(_):
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
  all_dev = gen_examples(dev_d, dev_q, dev_l, dev_a, config.batch_size)
  dev_acc = eval_acc()# TODO
  logging.info('Dev accuracy: %.2f %%' % dev_acc)
  best_acc = dev_acc

  if config.test_only:
      return
  
  # Training
  logging.info('-' * 50)
  logging.info('Start training..')
  train_d, train_q, train_l, train_a = utils.vectorize(train_examples, word_dict, entity_dict)
  assert len(train_d) == args.num_train
  all_train = gen_examples(train_d, train_q, train_l, train_a, config.batch_size)  
  start_time = time.time()
  n_updates = 0

  for epoch in range(config.num_epoches):
    np.random.shuffle(all_train)
    for idx, minibatch in enumerate(all_train):
      (doc, doc_mask, ques, ques_mask, labled, ans) = minibatch
      # TODO: training

if __name__ == '__main__':
  tf.app.run()
jn