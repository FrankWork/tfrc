# CNN/Daily Mail Reading Comprehension Task

Code for the paper:

[A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/pdf/1606.02858v2.pdf).


## Dependencies
* Python 3
* Tensorflow 1.0

## Datasets
* The two processed RC datasets:
    * CNN: [http://cs.stanford.edu/~danqi/data/cnn.tar.gz](http://cs.stanford.edu/~danqi/data/cnn.tar.gz) (546M)
    * Daily Mail: [http://cs.stanford.edu/~danqi/data/dailymail.tar.gz](http://cs.stanford.edu/~danqi/data/dailymail.tar.gz) (1.4G)

    The original datasets can be downloaded from [https://github.com/deepmind/rc-data](https://github.com/deepmind/rc-data) or [http://cs.nyu.edu/~kcho/DMQA/](http://cs.nyu.edu/~kcho/DMQA/).
    Our processed ones are just simply concatenation of all data instances and keeping document, question and answer only for our inputs.

* Word embeddings:
    * glove.6B.zip: [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

## Usage