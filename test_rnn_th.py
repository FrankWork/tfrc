import numpy as np
import theano
import theano.tensor as T
import lasagne

dropout_rate = 1
hidden_size = 3
num_layers = 1

def stack_rnn(l_emb, l_mask, num_layers, num_units,
              grad_clipping=10, dropout_rate=0.,
              bidir=True,
              only_return_final=False,
              name='',
              rnn_layer=lasagne.layers.GRULayer):
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

in_x1 = T.matrix('x1')
in_mask1 = T.matrix('mask1')


l_in1 = lasagne.layers.InputLayer((None, None), in_x1)
l_mask1 = lasagne.layers.InputLayer((None, None), in_mask1)

network = stack_rnn(l_in1, l_mask1, num_layers, hidden_size, name='d')
outputs = lasagne.layers.get_output(network, deterministic=True) * in_l
test_fn = theano.function([in_x1, in_mask1], outputs)

doc = np.ones([2, 3, 1], dtype=np.float32)
mask = np.ones([2, 3, 1], dtype=np.float32)
o = test_fn(doc, mask)




