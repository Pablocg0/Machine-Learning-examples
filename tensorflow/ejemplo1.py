import tensorflow as tf
import numpy as np
"""Example of Gradient descent in tensorflow """

trainX=np.linspace(-1,1,101) #vector with the features of size 101
trainY=3*trainX + np.random.randn(*trainX.shape)*0.33 #Vector with the output variables
X=tf.placeholder("float") #variable in tensorflow of type float
Y=tf.placeholder("float")
w = tf.Variable(0.0, name="weights")

init= tf.global_variables_initializer()


y_model=tf.mul(X,w) #linear Regression (theta' * x)

cost=(tf.pow(Y-y_model,2)) #costFunction (y - LinearRegression)^2
train_op= tf.train.GradientDescentOptimizer(0.01).minimize(cost)# GradientDescent with alpha= 0.01 and minimize costFunction

with tf.Session() as sess:
    sess.run(init) #run graph of tensorflow
    for i in range (100): #run GradientDescent for 100 iterations
        for (x, y) in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))
