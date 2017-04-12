import theano.tensor as T
import lasagne

hidden_size = 1
num_layers = 1
dropout_rate = 0.2
batch_size = 2
max_time = 3
embedding_size = 1
grad_clipping = 10
att_func = 'bilinear'

def stack_rnn(l_emb, l_mask, num_layers, num_units,
              grad_clipping=10, dropout_rate=0.,
              bidir=True,
              only_return_final=False,
              name='',
              rnn_layer=lasagne.layers.LSTMLayer):
    """
        Stack multiple RNN layers.
    """

    def _rnn(backwards=True, name=''):
        network = l_emb
        for layer in range(num_layers):
            if dropout_rate > 0:
                network = lasagne.layers.DropoutLayer(network, p=dropout_rate)
            c_only_return_final = only_return_final and (layer == num_layers - 1)
            network = rnn_layer(network, num_units,
                                grad_clipping=grad_clipping,
                                mask_input=l_mask,
                                only_return_final=c_only_return_final,
                                backwards=backwards,
                                name=name + '_layer' + str(layer + 1))
        return network

    network = _rnn(True, name)
    if bidir:
        network = lasagne.layers.ConcatLayer([network, _rnn(False, name + '_back')], axis=-1)
    return network

x = T.matrix('x')
mask = T.matrix('mask')

x_in = lasagne.layers.InputLayer((None, None), x)
mask_in = lasagne.layers.InputLayer((None, None), mask)

network = stack_rnn(x_in, mask_in, num_layers, hidden_size,
                                   grad_clipping=grad_clipping,
                                   dropout_rate=dropout_rate,
                                   only_return_final=(att_func == 'last'),
                                   bidir=True,
                                   name='d',
                                   rnn_layer=lasagne.layers.GRULayer)
train_fn = theano.function([x, mask], network)

train_fn(np.ones((batch_size, max_time)), np.ones((batch_size, max_time)))