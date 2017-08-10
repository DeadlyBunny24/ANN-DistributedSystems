import tensorflow as tf

def train_ANN(apps_number=1, mem_size=100, hidden_size=200, label_test=1, learning_rate=0.01, epochs=5):
    # Parameters:
    # mem_size      = mem_size/min_mem_partition_size
    # hidden_size   = Size of the hidden layer
    # label_test    = Temporary parameter to account for the missing label file
    # learning_rate = Learning rate of the Adagrad Optimizer
    # Epochs        = Number of Epochs for which to train with the whole dataset

    # Computation graph

    # Input dimensions are the HRC's [# of applications, Memory size/Smallest partition]
    inputs = tf.placeholder(tf.float32,[None,mem_size],'inputs')

    # Algorithm that yields the best memory partition. labels = {0,1,2}
    labels = tf.placeholder(tf.int32,name='labels')

    with tf.variable_scope('hidden_layer'):
        W = tf.get_variable('W',initializer= tf.truncated_normal([mem_size, hidden_size],stddev=1/tf.sqrt(tf.to_float(mem_size))))
        b = tf.get_variable('b',[hidden_size],initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('hidden_layer',reuse=True):
        hidden_outpt = tf.sigmoid(tf.matmul(inputs,W) + b)

    ksize = [1,apps_number,1,1]
    stride = [1,apps_number,1,1]
    embdd = tf.reshape(hidden_outpt, [1,apps_number,hidden_size,1], name='reshaping')
    max_pooling = tf.nn.max_pool(embdd, ksize=ksize, strides=stride,padding='SAME')
    max_pooling_t = tf.reshape(max_pooling,[1,hidden_size])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W',initializer= tf.truncated_normal([hidden_size, 3],stddev=1))
        b = tf.get_variable('b',[3],initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('softmax',reuse=True):
        logits = tf.sigmoid(tf.matmul(max_pooling_t,W)+b)

    predictions = tf.nn.softmax(logits, name='predictions')

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[labels], logits=logits, name='loss')

    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

    # Training

    # File queue
    file_list = tf.train.match_filenames_once("./traces/*.csv")
    file_name_queue = tf.train.string_input_producer(file_list, num_epochs=epochs)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(file_name_queue, mem_size)
    _, features = tf.decode_csv(value, record_defaults=[[0.0], [0.0]])
    pop_clear_queue = (file_name_queue.dequeue(), reader.reset())

    with tf.Session() as sess:
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(epochs):
            input_test = [sess.run(features) for j in range(apps_number)]
            _,loss_value = sess.run([train_step, loss], feed_dict={inputs:input_test, labels:label_test})
            print 'Loss of epoch {}: {}'.format(i,loss_value)
            sess.run(pop_clear_queue)

        coord.request_stop()
        coord.join(threads)


    # Testing
