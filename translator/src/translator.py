import tensorflow as tf
import os


def predict(word2int, es_word2int, int2word, es_int2word, en_max, batch_size,
            path, input_sentence):
    """
    Description:
        Using the model as a reference, translates given english text to Spanish
    """
    text = input_sentence.lower()
    text = source_to_seq(text.split(), word2int, en_max)
    checkpoint = path + "/best_model.ckpt"

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name(
            'source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name(
            'target_sequence_length:0')

        # Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          target_sequence_length: [len(
                                              text)] * batch_size,
                                          source_sequence_length: [len(
                                              text)] * batch_size})[0]

    pad = word2int["<PAD>"]
    print('Original Text:', input_sentence)
    print('\nSource')
    print('  Word Ids:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([int2word[i] for i in text])))

    print('\nTarget')
    print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(
        " ".join([es_int2word[i] for i in answer_logits if i != pad])))


def source_to_seq(text, word2int, en_max):
    '''Prepare the text for the model'''
    sequence_length = len(text)
    return [word2int.get(word, word2int['<UNK>']) for word in
            text] + [word2int['<PAD>']]*(sequence_length-len(text))
