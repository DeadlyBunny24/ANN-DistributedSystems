import tensorflow as tf
import numpy as np

def train_ANN(mem_size=12702, hidden_size=200, learning_rate=0.01, epochs=5):

    # Parameters ===============================================================

    # mem_size      = mem_size/min_mem_partition_size
    # hidden_size   = Size of the hidden layer
    # learning_rate = Learning rate of the Adagrad Optimizer
    # Epochs        = Number of Epochs for which to train with the whole dataset


    # Computational graph ======================================================

    # Input dimensions are the HRC's [#_of_applications, #_of_memory_partitions]
    inputs = tf.placeholder(tf.float32,[None,mem_size],'inputs')

    # Labels correpond to the Algorithm that yields the best memory partition.
    # labels = {0,1,2}
    labels = tf.placeholder(tf.int32,name='labels')

    # Hidden layer
    with tf.variable_scope('hidden_layer'):
        W = tf.get_variable('W',initializer= tf.truncated_normal([mem_size, hidden_size],
                                stddev=1/tf.sqrt(tf.to_float(mem_size))))
        b = tf.get_variable('b',[hidden_size],
                                initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('hidden_layer',reuse=True):
        hidden_outpt = tf.sigmoid(tf.matmul(inputs,W) + b)

    # Max-pooling layer
    with tf.name_scope('pooling_layer'):
        max_pool_h = tf.reduce_max(hidden_outpt,axis=0,keep_dims=True)

    # Softmax layer
    with tf.variable_scope('output'):
        W = tf.get_variable('W',initializer= tf.truncated_normal([hidden_size, 3],
                                stddev=1/tf.sqrt(tf.to_float(hidden_size))))
        b = tf.get_variable('b',[3],initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('output',reuse=True):
        logits = tf.matmul(max_pool_h,W)+b

    # Prediction of the model. Meant to be used on accuracy calculation.
    predictions = tf.nn.softmax(logits, name='predictions')

    # Loss function
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[labels],
                                                          logits=logits,
                                                          name='loss')

    # Optimizer
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)


    # Training =================================================================


    directory = 'ms_{m:03d},hs_{h:03d},lr_{l:.0E},ep_{e:03d}'.format(
        m=mem_size,
        h=hidden_size,
        l=learning_rate,
        e=epochs,
    )

    # File writer to save metadata about training
    writer = tf.summary.FileWriter('./tmp/'+directory)

    # Saves the model to directory
    saver = tf.train.Saver()

    # File queue

    # Creates a list of strings that match the pattern "./traces/*.csv"
    file_list = tf.train.match_filenames_once("./traces/*.csv")

    # Creates a queue of strings. Each element is a filename from filelist
    # This queue is repeated Epochs number of times
    file_name_queue = tf.train.string_input_producer(file_list,
                                                    num_epochs=epochs)


    with tf.Session() as sess:
        # Intializes global and local variables.
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # Starts the input pipeline threads.
        # TODO: Update the input pipeline to use the Dataset API.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while(True):
                # Obtains a filename from a queue of files.
                filename_pop = sess.run(file_name_queue.dequeue())

                # Process the csv file
                input_ = np.genfromtxt(filename_pop, delimiter=',')

                # Transposes the input: [#_mem_partitions, #_applications] ->
                #                       [#_applpications,#_mem_partitions]
                input_t = np.transpose(input_[:,1:])

                # TODO: Define the input format of labels.
                # This line is implemented to account for the lack of labels
                # in the csv files.
                # This is fed as the correct label for every iteration.
                # Each iteration requires a scalar label.
                label_t = 1

                _,loss_value = sess.run([train_step, loss],
                                    feed_dict={inputs:input_t, labels:label_t})

                print loss_value

        except tf.errors.OutOfRangeError:
            # Merges all threads
            coord.request_stop()
            coord.join(threads)

            # Creates an events file for Tensorboard
            writer.add_graph(sess.graph)

            # Creates a checkpoint file that contains the trained model.
            # This file can later be used to restore the model.
            saver.save(sess, "./tmp/{}/directory.ckpt".format(directory))
            print 'Done training!'
        else:
            # Merges all threads
            coord.request_stop()
            coord.join(threads)
            print 'Done training!'
