import tensorflow as tf
import numpy as np

def train_ANN(mem_size=12702, hidden_size=200, learning_rate=0.01, epochs=5):
    # Parameters:
    # mem_size      = mem_size/min_mem_partition_size
    # hidden_size   = Size of the hidden layer
    # learning_rate = Learning rate of the Adagrad Optimizer
    # Epochs        = Number of Epochs for which to train with the whole dataset


    # Computation graph

    # Input dimensions are the HRC's [# of applications, Memory size/Smallest partition]
    inputs = tf.placeholder(tf.float32,[None,mem_size],'inputs')

    # Algorithm that yields the best memory partition. labels = {0,1,2}
    labels = tf.placeholder(tf.int32,name='labels')

    # Hidden layer
    with tf.variable_scope('hidden_layer'):
        W = tf.get_variable('W',initializer= tf.truncated_normal([mem_size, hidden_size],stddev=1/tf.sqrt(tf.to_float(mem_size))))
        b = tf.get_variable('b',[hidden_size],initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('hidden_layer',reuse=True):
        hidden_outpt = tf.sigmoid(tf.matmul(inputs,W) + b)

    # Max pool layer
    max_pool_h = tf.reduce_max(hidden_outpt,axis=0,keep_dims=True)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W',initializer= tf.truncated_normal([hidden_size, 3],stddev=1))
        b = tf.get_variable('b',[3],initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('softmax',reuse=True):
        logits = tf.sigmoid(tf.matmul(max_pool_h,W)+b)

    # Prediction by the model
    predictions = tf.nn.softmax(logits, name='predictions')

    # Loss function
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[labels], logits=logits, name='loss')

    # Optimizer
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)


    # Training

    # File queue
    file_list = tf.train.match_filenames_once("./traces/*.csv")
    file_name_queue = tf.train.string_input_producer(file_list, num_epochs=epochs)


    with tf.Session() as sess:
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while(True):
                filename_pop = sess.run(file_name_queue.dequeue())
                input_ = np.genfromtxt(filename_pop, delimiter=',')
                input_t = np.transpose(input_[:,1:])
                label_t = 1 # To account for the missing labels
                _,loss_value = sess.run([train_step, loss], feed_dict={inputs:input_t, labels:label_t})
                print loss_value
        except Exception, e:
            print 'Done training!'
        else:
            coord.request_stop()
            coord.join(threads)
