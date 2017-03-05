import tensorflow as tf
import numpy as np
import pandas as pd

''' Example the neuralMetwork in tensorflow  for the logical connective AND'''

inputs = tf.placeholder("float", name="Inputs"); #Variable of type float
data = np.array([[1,0],[1,1],[0,1],[1,0]]); #data for the values the true table

#Define outpus
one = lambda: tf.constant(1.0);
zero = lambda: tf.constant(0.0);

with tf.name_scope('weights'):
    #define weights and slant
    weights = tf.placeholder("float", name="weights");
    slant = tf.placeholder("float", name="slant");

with tf.name_scope("activation"):
    #Activation function
    activation = tf.reduce_sum(tf.add(tf.matmul(inputs,weights),slant));

with tf.name_scope("neuron"):
    #we define the neuron
    def neuron():
        return tf.case([(tf.less(activation, 0.0),zero)],default= one);

    a = neuron(); #output

logs_path = '/neuron'



#Run Session
with tf.Session() as sess:
    #We assemble the graph
    summary_writer = tf.train.SummaryWriter(logs_path,graph=sess.graph);

    #to assemble true table
    x_1 = []
    x_2 = []
    out = []
    act = []
    for i in range(len(data)):
        t = data[i].reshape(1, 2);
        output, activ = sess.run([a, activation], feed_dict={inputs: t,
                                        weights:np.array([[1.],[1.]]),
                                        slant: -1.5});
        #to assemble true table in DataFrame
        x_1.append(t[0][0]);
        x_2.append(t[0][1]);
        out.append(output);
        act.append(activ);
    table_info = np.array([x_1, x_2, act, out]).transpose();
    table = pd.DataFrame(table_info, columns=['x1', 'x2', 'f(x)', 'x1 AND x2']);
print table
