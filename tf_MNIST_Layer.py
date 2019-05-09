import tensorflow as tf
#number 1 to 10 data
from tensorflow.examples.tutorials.mnist import input_data
#MNIST_DATA='/Users/Adcc/python_workdir/workdir/MNIST_data'
#mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)
print("MNIST data is loading.......")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print("MNIST data is loaded........")

#add one or more layers and return the outputs
def add_layer(inputs,in_size,out_size,activation_function=None):
    #add one more layer and return  the output of this layer
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    #print("prediction:",y_pre)
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))#check the max value position
    print("correct_pridiction",sess.run(correct_prediction,feed_dict={ys:y_pre}))
    tf_cast=tf.cast(correct_prediction, tf.float32)
    print("tf_cast:",sess.run(tf_cast))
    accuracy=tf.reduce_mean(tf_cast)
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    print("result",result)
    return result


#define placeholder for inputs to network
xs=tf.placeholder(tf.float32,[None,784])#28X28 pixels
ys=tf.placeholder(tf.float32,[None,10])#10 numbers

#add output layer
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
#softax is to use classificstion

#the error between prediction and real data
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                            reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#important step
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(200)
    # print("Batch X:",batch_xs)
    # print("Batch Y:",batch_ys)
    tmp_train=sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    #print("Train output:",tmp_train)

    if i%100==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
