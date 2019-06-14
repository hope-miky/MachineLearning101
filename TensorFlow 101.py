#!/usr/bin/env python
# coding: utf-8

# # TensorFlow 101

# ####  Tensor ( multidimensional data) flowing through a Graph!

# * Opensource Machine Learning Library
# * Mainly for Deep Learning
# * For both Research and Production

# # ድግስFlow

# * Two stages
# 
#     * Coocking using the Recipe
#     
#     * Serving the Guests

# ### What about TensorFlow?

# * Same as ድግስFlow, it's just...
#     * Graph
#     * Session

# ### What is Graph?

# * A collection of Nodes or Coputations.
# * Defined in high-level Languages ( e.g. Python)
# * Executed on available low level devices (e.g CPU)

# ### What is Session?

# * Operation Excution 
# * Tensor evaluation

# # Let's CODE!

# ## Softmax ( Multinomial Logistic Regression)

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ####  Preparing the Data

# In[2]:


# Downloading or just importing the MNIST dataset from tensorflow examples
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


# Extracting the data and loading it into a variable
mnist = input_data.read_data_sets("Datas/MNIST_data/", one_hot=True)


# In[4]:


# quantity of the images in the dataset
mnist.train.num_examples, mnist.test.num_examples


# In[5]:


# Randomly selecting an image to display and see what is  looks like
random_image = mnist.train.images[100].reshape(28,28)


# In[6]:


# Displaying an image on the axes
plt.imshow(random_image, cmap="gist_gray")


# In[7]:


# Checking wether the data is notmalized or not
random_image.min(), random_image.max()


# ## Let's build the graph!

# Multinomial Logistic Regression interms of mathematical equation is:
# 
#      y = Wx + b

# * x = the image data to be feed in the session
# * W = the wight 
# * b = the bias

# In[8]:


# placeholder for the training dataset
# Vectorized_image_Data.shape = 28*28 --. 784
x = tf.placeholder(tf.float32, shape=[None, 784])


# In[9]:


# declaring and initializing Variables into zero!
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[10]:


# Creating the Logit or Score of the computation
y = tf.matmul(x,w) + b


# In[11]:


# Loss Function
y_true = tf.placeholder(tf.float32, shape=[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))


# In[12]:


# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(cross_entropy)


# ##  Let's Create the Session!

# In[13]:


# Initializing all variables
init = tf.global_variables_initializer()


# In[14]:


with tf.Session() as sess:
    
    sess.run(init)
    
    for steps in range(100000):
        
        x_batch, y_batch = mnist.train.next_batch(100)
        
        sess.run(train, feed_dict={x:x_batch, y_true:y_batch})
        
    # Evaluate the model
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print(sess.run(accuracy, feed_dict = {x:mnist.test.images, y_true:mnist.test.labels}))


# # Done!
