import data_parsing as data
import word2vec as w2v
import seq2seq as seq
import translator as translate

import numpy as np
import pickle
import os


def main():
    # Set up source and data paths
    work_path = os.getcwd()
    os.chdir("..")
    data_path = os.getcwd() + '/data'
    os.chdir(data_path)

    # Parameters
    vocab_size = 5000
    batch_size = 64
    embed_dim = 32
    epochs = 500

    userAnswer = input("Would you like to retrain the neural network? (y/N)")

    if "y" is userAnswer:
        train_seq2seq = 1
    elif "N" is userAnswer:
        train_seq2seq = 0

    if(train_seq2seq):
        en_words, en_max = data.dev_source()
        es_words, es_max = data.dev_target()
        #print(en_words[:2])

        en_data, en_count, word2int, int2word = data.build_dataset(en_words, vocab_size)
        es_data, es_count, es_word2int, es_int2word = data.build_dataset(es_words, vocab_size)
        #print(en_data[:2])
        del en_words, es_words

        seq.seq2seq_train(en_data, es_data, word2int, es_word2int, en_max,
                            es_max, data_path, batch_size, embed_dim, epochs)

        print("Saving checkpoints to data directory.")
        pickle.dump(en_data, open("en_data.p", "wb"))
        pickle.dump(es_data, open("es_data.p", "wb"))
        pickle.dump(word2int, open("word2int.p", "wb"))
        pickle.dump(es_word2int, open("es_word2int.p", "wb"))
        pickle.dump(int2word, open("int2word.p", "wb"))
        pickle.dump(es_int2word, open("es_int2word.p", "wb"))
        pickle.dump(en_max, open("en_max.p", "wb"))
        pickle.dump(es_max, open("es_max.p", "wb"))
        pickle.dump(batch_size, open("batch_size.p", "wb"))

    else:
        print("Loading checkpoints from data directory.")
        en_data = pickle.load(open("en_data.p", "rb"))
        es_data = pickle.load(open("es_data.p", "rb"))
        word2int = pickle.load(open("word2int.p", "rb"))
        es_word2int = pickle.load(open("es_word2int.p", "rb"))
        int2word = pickle.load(open("int2word.p", "rb"))
        es_int2word = pickle.load(open("es_int2word.p", "rb"))
        en_max = pickle.load(open("en_max.p", "rb"))
        es_max = pickle.load(open("es_max.p", "rb"))
        batch_size = pickle.load(open("batch_size.p", "rb"))

        print("Let's translate! Input text in English and we will translate to Spanish!")
        en_in = input("Enter desired text: ")

        translate.predict(word2int, es_word2int, int2word, es_int2word,
                              en_max, batch_size, data_path, en_in)




if __name__ == "__main__":
    main()
