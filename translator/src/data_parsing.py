import re
import os
import collections


# Set of words to remove from parse data
stopwords = ['but', 'there', 'about', 'they', 'an', 'be', 'for', 'do', 'its',
             'of', 'is', 's', 'am', 'or', 'as', 'the', 'are', 'we', 'don',
             'nor', 'this', 'to', 'at', 'so', 'i', 'll', 't', 'if', 'a', 'by',
             'it','and', 'in', 'that', 'on', 'you', 'who']

es_stopwords = ['el', 'la', 'los', 'un', 'una', 'unas', 'uno', 'sobre', 'todo',
                'tras', 'otro', 'de', 'del', 'que', 'en', 'y', 'a', 'las',
                'se', 'no', 'es', 'para', 'por', 'con']


# http://www.statmt.org/wmt16/translation-task.html
europarl_en = 'europarl-v7.es-en.en'
europarl_es = 'europarl-v7.es-en.es'
# http://www.statmt.org/wmt15/quality-estimation-task.html
source = 'dev.source'
target = 'dev.target'

europarl_sentences = 1965734
wmt15_sentences = 11271


def build_dataset(sentences, n_words):
    """
    Description:
        Builds 2 lists and 2 dictionaries in a specific order so we can use
        for tensorflow neural networks

    Keyword Arguments:
        sentences - list of all words in original sentence order
        n_words - number of words to consider

    Notes:
        'UNK' word is not in our vocabulary
        'EOS' tells us where the sentence ends
        'PAD' is used during training to help batch sentences
        'GO'  is so the decoder knows when to start generating output

    Returns:
        data - list that takes comments in the original order and replaces them
        with a number signifying how often they're encountered
        count - list of tuples containing most popular words with keys for each
        unique word
        dictionary - dictionary in random order with keys for each unique word
        and the value being its ranking in terms of popularity
        reverse - reversal of dictionary
    """
    count = [['<UNK>', 0], ['<EOS>', 1], ['<PAD>', 2], ['<GO>', 3]]
    flattened = sum(sentences, [])
    count.extend(collections.Counter(flattened).most_common(n_words-4))
    dictionary = dict()
    for word, _ in count:
        if(word == "<EOS>"):
            dictionary[word] = 1
        else:
            dictionary[word] = len(dictionary)
    word_data = list()
    sentence_data = list()
    unk_count = 0
    for sentence in sentences:
        for word in sentence:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            word_data.append(index)
        sentence_data.append(list(word_data))
        word_data = []
    count[0][1] = unk_count
    reverse = dict(zip(dictionary.values(), dictionary.keys()))
    return sentence_data, count, dictionary, reverse


def dev_target():
    """
    Description:
        parses dev.target file

    Return:
        es_sentences - list of sentences in spanish
        max_length - max sentences length found
    """
    es_sentences = []
    lines = []
    i = 0
    with open("dev.target") as f:
        for line in f:
            i = i + 1
            if i > wmt15_sentences:
                break

            line = re.sub(r'[^a-zA-Zá-úÁ-Úñ]+', ' ', line)
            text = line.lower().split()
            text.append("<EOS>")
            # text = [word for word in text if word not in es_stopwords]
            lines.append(len(text))
            es_sentences.append(text)

        max_length = max(lines)
        return es_sentences, max_length

def dev_source():
    """
    Description:
        parses dev.source file

    Return:
        en_sentences - list of sentences in original order in english
        max_length - max sentence length found
    """
    en_sentences = []
    lines = []
    i = 0
    with open("dev.source") as f:
        for line in f:
            i = i + 1
            if i > wmt15_sentences:
                break

            line = re.sub(r"n \'t", " not", line)
            line = re.sub(r"n\' t", " not", line)
            line = re.sub(r"\'re", "are", line)
            line = re.sub(r"\' ve", "have", line)
            line = re.sub(r"\'ve", "have", line)
            line = re.sub(r"\'s", "is", line)
            line = re.sub(r"\'ll", "will", line)
            line = re.sub(r"\' ll", "will", line)
            line = re.sub(r"\'m", "am", line)
            line = re.sub(r'[^a-zA-Z]+', ' ', line)
            text = line.lower().split()
            # text.append("<EOS>")
            # text = [word for word in text if word not in stopwords]
            lines.append(len(text))
            en_sentences.append(text)

        max_length = max(lines)
        return en_sentences, max_length

def parse_europarl():
    """
    Description:
        Cleans up Europarl sentences and add characters necessary
        for seq2seq processing such as <EOS> and <PAD>

    Note:
        both files contain 1,965,734 sentences

    Returns:
        en_words - list of words in original order from sentences
    """
    en_words = []
    i = 0
    lines = []
    with open(europarl_en) as f:
        for line in f:
            lines.append(len(line))
            i = i + 1
            if i > europarl_sentences:
                break

            # convert words to lower case and split them
            text = line.lower()

            # clean text
            text = re.sub(r"[()]", '', text)
            text = re.sub(r"\\n", '', text)
            text = re.sub(r"what\' s", "what is ", text)
            text = re.sub(r"\' s", " ", text)
            text = re.sub(r"\' ve", " have ", text)
            text = re.sub(r"can' t", "cannot ", text)
            text = re.sub(r"n\' t", " not ", text)
            text = re.sub(r"i\' m", "i am ", text)
            text = re.sub(r"\' re", " are ", text)
            text = re.sub(r"\' d", " would ", text)
            text = re.sub(r"\' ll", " will ", text)
            text = re.sub(r",", "", text)
            text = re.sub(r"\.", "", text)
            text = re.sub(r"!", " !", text)
            text = text.split()
            text = [word for word in text if word not in stopwords]
            en_words.extend(text)

    return en_words


def parse_europarl_es():
    """
    Description:
        Parses the europarl file in spanish.
        Seperates Sentences with <EOS> and breaks them into a giant list where
        each element is a word

    Returns:
        list of words in original order from sentences
    """
    es_sentences = []
    lines = []
    i = 0
    with open(europarl_es) as f:
        for line in f:
            lines.append(len(line))
            i = i + 1
            if i > europarl_sentences:
                break

            line = line.lower()

            line = re.sub(r'[()]', '', line)
            line = re.sub(r'[^a-zA-Zá-úÁ-Úñ]+', ' ', line)
            line = line.split()
            text = [word for word in line if word not in es_stopwords]
            es_sentences.extend(text)

    return es_sentences