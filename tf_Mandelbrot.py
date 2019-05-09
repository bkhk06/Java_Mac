####Zn+1=Zn^2+C
####
#set the grid for the MB plot, and define c as complex numbers
#the corresponding tf object is zs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
Y,X = np.mgrid[-1.3:1.3:0.005,-2:1:0.005]
Z = X+1j*Y
c = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(c)
ns = tf.Variable(tf.zeros_like(c,tf.float32))


sees = tf.InteractiveSession()
tf.global_variables_initializer().run()


zs_ = zs*zs+c
not_diverged = tf.abs(zs_)<10
step = tf.group(zs.assign(zs_),ns.assign_add(tf.cast(not_diverged,tf.float32)))

for i in range(125):step.run()


plt.imshow(ns.eval())
plt.show()







