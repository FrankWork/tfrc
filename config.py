import tensorflow as tf


# Basics
tf.app.flags.DEFINE_boolean("debug", True,
                            "run in the debug mode.")
tf.app.flags.DEFINE_boolean("test_only", False,
                            "no need to run training process.")

# Data files
tf.app.flags.DEFINE_string("data_path", "data/", "Data directory")
tf.app.flags.DEFINE_string("embedding_file", "glove/glove.6B.100d.txt", 
                           "embedding file")
tf.app.flags.DEFINE_string("train_file", "cnn/train.txt", "training file")
tf.app.flags.DEFINE_string("dev_file", "cnn/dev.txt", "Development file")
tf.app.flags.DEFINE_string("test_file", "cnn/test.txt", "Test file")
tf.app.flags.DEFINE_string("log_file", None, "Log file")
tf.app.flags.DEFINE_string("save_path", "model/", "save model here")


# Model details
tf.app.flags.DEFINE_integer("hidden_size", "128", "Hidden size of RNN units")
tf.app.flags.DEFINE_integer("num_layers", "1", "Number of RNN layers")

# Optimization details
tf.app.flags.DEFINE_integer("batch_size", "32", "Batch size")
tf.app.flags.DEFINE_integer("num_epoches", "30", "Number of epoches")
tf.app.flags.DEFINE_integer("eval_iter", "100", 
                            "Evaluation on dev set after K updates")
tf.app.flags.DEFINE_float("dropout_rate", 0.2, "Dropout rate.")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("grad_clipping", 10., "Gradient clipping.")


FLAGS = tf.app.flags.FLAGS