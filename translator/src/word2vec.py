import numpy as np
import tensorflow as tf
import collections

batch_size = 128
iterations = 10000
learn_rate = 0.03
data_index = 0


def batch_data(words, size, num_skips, skip_window):
    """
    Description:
        Generates mini-batches to use during training.
        With batch and context you can then create
        proper Word2vec training code

    Keyword arguments:
        words - A list of individual words.

        batch_size - Size of current batch/words. Must be divisible by
        num_skips.

        num_skips - How many times to reuse an input

        skip_window - How many words to consider left and right

    Notes:
        batch and context are arrays used for one-hot vector association

    Returns:
        batch - Consists of input words which are matched with each
        context word in their context
        context - Random associated context words within the gram
        as the labels to predict
    """
    global data_index
    assert size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(size,), dtype=np.int32)
    context = np.ndarray(shape=(size, 1), dtype=np.int32)
    # span = 2(skip window) + 1
    span = 2 * skip_window + 1
    # Double sided data structure
    buffer = collections.deque(maxlen=span)
    # Buffer holds a max of span elements and will be shifting right.
    for _ in range(span):
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)
    # First target is selected is the word at the center of the span.
    # Other words are randomly selected from the span.
    # Make sure input is not selected for context.
    for y in range(size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for z in range(num_skips):
            while target in targets_to_avoid:
                target = np.random.random_integers(0, span - 1)
            targets_to_avoid.append(target)
            # this is the input word
            batch[y * num_skips + z] = buffer[skip_window]
            # these are the context words
            context[y * num_skips + z, 0] = buffer[target]
        buffer.append(words[data_index])
        data_index = (data_index + 1) % len(words)
    # Backtrack a little bit to avoid skipping words
    data_index = (data_index + len(words) - span) % len(words)
    return batch, context


def train_context(size, dim, num_iterations, words,
                  word2int, int2word):
    """ Function that trains context per word.
        So for a given word it will return relevant words
        based on the context of sentences.
    """
    # placeholders for x_train and y_train
    x = tf.placeholder(tf.float32, shape=(None, size))
    y_label = tf.placeholder(tf.float32, shape=(None, size))

    # weights connecting the input layer to the hidden layer
    w1 = tf.Variable(tf.random_normal([size, dim]))
    # bias
    b1 = tf.Variable(tf.random_normal([dim]))

    # weights connecting hidden layer to the output layer
    w2 = tf.Variable(tf.random_normal([dim, size]))
    # bias
    b2 = tf.Variable(tf.random_normal([size]))

    # calculate output of the hidden layer
    hidden_layer = tf.add(tf.matmul(x, w1), b1)
    # calculate output using softmax activated output layer
    prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, w2), b2))

    # define cost function that we use to train model
    loss = tf.reduce_mean(
        -tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

    # define the training step with optimization
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learn_rate).minimize(loss)

    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    average_loss = 0
    for step in range(num_iterations+1):
        batch_inputs, batch_labels = batch_data(words, size=size,
                                                num_skips=2, skip_window=5)
        x_train, y_train = matrix_creation(batch_inputs, batch_labels,
                                           word2int, int2word)
        feed_dict = {x: x_train, y_label: y_train}

        _, loss_val = sess.run(
            [optimizer, loss],
            feed_dict=feed_dict
        )
        average_loss += loss_val
        if step % (num_iterations/5) == 0:
            if step > 0:
                average_loss /= (num_iterations/5)
            print('Average loss, step', step, ': ', average_loss)
            average_loss = 0

    vectors = sess.run(w1 + b1)
    return vectors



def matrix_creation(batch, context, word2int, int2word, size):
    """
    Description:
        Creates matrix for word2vec training

    Keyword Arguments:
        :param batch:
        :param context:
        :param word2int:
        :param int2word:

    Returns:
        x_train - input matrix of words represented as integers
        y_train - output matrix of words representeed as integers
    """
    word_pairs = []
    for i in range(batch_size):
        checked = int2word[batch[i]]
        relevant = int2word[context[i, 0]]
        word_pairs.append([checked, relevant])

    # create matrices using integers represented as words
    x_train = []    # input words
    y_train = []    # output words
    for pair in word_pairs:
        x_train.append(to_one_hot(word2int[pair[0]], size))
        y_train.append(to_one_hot(word2int[pair[1]], size))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    return x_train, y_train


def to_one_hot(data_point, size):
    """
    Description:
        Function to convert list to one hot vector

    Keyword Arguments:
        data_point - single 1 in the vector
        size - size of vector is the size of your vocabulary

    Returns:
        temp - one-hot vector as numpy array
    """
    temp = np.zeros(size)
    temp[data_point] = 1
    return temp


data_index = 0


def find_closest(index, vectors):
    """
    Description:
        Finds the closest vector to a given one using
        the euclidean distance formula

    Note:
        This is really showing what word is the closest related to a given word

    Returns:
        min_index - index of closest vector
    """
    min_distance = 100000  # positive infinity
    min_index = 0

    query_vec = vectors[index]

    for index, vector in enumerate(vectors):
        if euclidean_dist(vector,
                          query_vec) < min_distance and not np.array_equal(
                vector, query_vec):
            min_distance = euclidean_dist(vector, query_vec)
            min_index = index

    return min_index



def euclidean_dist(vec1, vec2):
    """
    Description:
        Euclidean function used for cleaner code
    Returns:
        Euclidean distance between two vectors
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))
