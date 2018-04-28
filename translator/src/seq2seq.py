import os
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense

# number of steps until training loss check
display_step = 2

# learning rate for the sequence to sequence model
learning_rate = 0.03


def seq2seq_train(en_source, es_target, word2int, es_word2int, en_max,
                  es_max, path, batch_size, embed_dim, epochs):
    """
    Description:
        Defines loss and optimizer then trains seq2seq model
    """
    train_graph = tf.Graph()
    with train_graph.as_default():
        (input_data, targets, learn_rate, target_sequence_length,
         max_target_sequence_length, source_sequence_length) = model_inputs()

        training_decoder_output, inference_decoder_output = seq2seq_model(
            input_data, targets, target_sequence_length,
            max_target_sequence_length, source_sequence_length, embed_dim,
            rnn_size=50, es_word2int=es_word2int, batch_size=batch_size)

        training_logits = tf.identity(training_decoder_output.rnn_output,'logits')
        inference_logits = tf.identity(inference_decoder_output.sample_id,
                                       name='predictions')


        masks = tf.sequence_mask(target_sequence_length,
                                 max_target_sequence_length,
                                 dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learn_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for
                                grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

    # Split data to training and validation sets
    train_source = en_source[batch_size:]
    train_target = es_target[batch_size:]
    valid_source = en_source[:batch_size]
    valid_target = es_target[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths,
     valid_sources_lengths) = next(
        get_batches(valid_target, valid_source, batch_size,
                    word2int['<PAD>'],
                    es_word2int['<PAD>']))

    checkpoint = path + "/best_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths,
                          sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                                word2int['<PAD>'],
                                es_word2int['<PAD>'])):

                # Training step
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     learn_rate: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})

                # Debug message updating us on the status of the training
                if batch_i % display_step == 0 and batch_i > 0:
                    # Calculate validation cost
                    validation_loss = sess.run(
                        [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         learn_rate: learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths})

                    print(
                        'Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                        .format(epoch_i,
                                epochs,
                                batch_i,
                                len(train_source) // batch_size,
                                loss,
                                validation_loss[0]))

        # Save Model
        os.chdir(path)
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')


def get_batches(targets, sources, size, source_pad_int, target_pad_int):
    """
    Description:
        Batch targets, sources, and the lengths of their sentences together

    """
    for batch_i in range(0, len(sources) // size):
        start_i = batch_i * size
        sources_batch = sources[start_i:start_i + size]
        targets_batch = targets[start_i:start_i + size]
        pad_sources_batch = np.array(
            pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(
            pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield (pad_targets_batch, pad_sources_batch, pad_targets_lengths,
               pad_source_lengths)

def pad_sentence_batch(sentence_batch, pad_int):
    """
    Description:
        <PAD> sentences so that each sentence of a batch has the same length

    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for
            sentence in sentence_batch]


def seq2seq_model(input_data, targets, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  embed_size, rnn_size, es_word2int, batch_size):

    encoder_output, enc_state = encoding_layer(input_data,
                                  rnn_size,
                                  source_sequence_length,
                                  len(es_word2int),
                                  embed_size)

    dec_input = process_decoder_input(targets, es_word2int, batch_size)

    training_decoder_output, inference_decoder_output = decoding_layer(
        es_word2int, embed_size, rnn_size, target_sequence_length,
        max_target_sequence_length, enc_state, dec_input, batch_size,
        source_sequence_length, encoder_output)

    return training_decoder_output, inference_decoder_output


def decoding_layer(es_word2int, decoding_embedding_size, rnn_size,
                   target_length, max_target_length, enc_state, dec_input,
                   batch_size, source_sequence_length, encoder_output):
    """
    Description:
        1- Process decoder inputs
        2- Set up the decoder components
            - Embedding
            - Decoder cell
            - Dense output layer
            - Training decoder
            - Inference decoder
    """
    # Decoder Embedding
    target_vocab_size = len(es_word2int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,
                                                    decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # construct the decoder cell
    dec_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    # Dense layer to translate the decoders output at each time
    # step into a choice from the target vocab
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(
                             mean=0.0, stddev=0.1))

    # Set up a training decoder and an inference decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=dec_embed_input, sequence_length=target_length
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            training_helper,
                                                            enc_state,
                                                            output_layer)

        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                                        training_decoder, impute_finished=True,
                                        maximum_iterations=max_target_length)[0]

    # Inference Decoder
    #Reuses the same parameters trained by the training process
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([es_word2int['<GO>']],
                                           dtype=tf.int32), [batch_size],
                               name='start_tokens')

    # helper for the inerence process
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        dec_embeddings, start_tokens, es_word2int['<EOS>'])

    # basic decoder
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                     inference_helper,
                                                     enc_state,
                                                     output_layer)

    # Perform dynamic decoding using the decoder
    inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
     inference_decoder, impute_finished=True,
     maximum_iterations=max_target_length)[0]

    return training_decoder_output, inference_decoder_output


def process_decoder_input(target_data, word2int, size):
    """
    Description:
        Remove the last word id from each batch and concat the <GO>
        to the beginning of each batch

    """
    ending = tf.strided_slice(target_data, [0, 0], [size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([size, 1], word2int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(input_data, rnn_size, source_sequence_length,
                   source_vocab_size, embed_size):
    """
    Description:
        The first bit of the model we'll build is the encoder.
        Here, we'll embed the input data, construct our encoder,
        then pass the embedded data to the encoder.
    """

    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,
                                                       source_vocab_size,
                                                       embed_size)

    enc_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    enc_output, enc_state = tf.nn.dynamic_rnn(
                                        enc_cell, enc_embed_input,
                                        sequence_length=source_sequence_length,
                                        dtype=tf.float32)
    return enc_output, enc_state


def model_inputs():
    """
    Description:
        Define tensorflow variables
    """
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learn_rate = tf.placeholder(tf.float32, name='learn_rate')

    target_sequence_length = tf.placeholder(tf.int32, (None,),
                                            name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length,
                                               name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,),
                                            name='source_sequence_length')
    return (input_data, targets, learn_rate, target_sequence_length,
            max_target_sequence_length, source_sequence_length)
