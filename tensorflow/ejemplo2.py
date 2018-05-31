import tensorflow as tf

"""Example of lineal model in tensorflow"""

W = tf.Variable([.3], tf.float32) #Variable in tensorflow with the value = .3
b = tf.Variable([-.3], tf.float32)#Variable in tensorflow with the value =  -.3
x = tf.placeholder(tf.float32)#Variable in tensorflow
linear_model = W * x + b

init= tf.global_variables_initializer() #Initialization of global values
sess= tf.Session()
sess.run(init)#run graph 
print(sess.run(linear_model, {x:[1,2,3,4]})) #Evaluation of the model with the values  x= 1,2,3,4
