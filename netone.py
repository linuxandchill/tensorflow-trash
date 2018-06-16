import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

h1_nodes = 500
h2_nodes = 500
h3_nodes = 500
n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.plcaeholder('float')

def model(data):
    hl_1 = {'weights': tf.Variable(tf.random_normal([784, h1_nodes])),
            'biases': tf.Variable(tf.random_normal(h1_nodes))}
    hl_2 = {'weights': tf.Variable(tf.random_normal([h1_nodes, h2_nodes])),
            'biases': tf.Variable(tf.random_normal(h2_nodes))}
    hl_3 = {'weights': tf.Variable(tf.random_normal([h2_nodes, h3_nodes])),
            'biases': tf.Variable(tf.random_normal(h3_nodes))}
    o_layer = {'weights': tf.Variable(tf.random_normal([h3_nodes, n_classes])),
            'biases': tf.Variable(tf.random_normal(n_classes))}

    layer_1 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    layer_1 =  tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(data, hl_2['weights']), hl_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(data, hl_3['weights']), hl_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3, o_layer['weights'] + o_layer['biases'])

    return output

def train_model(input): 


train_model(x)

