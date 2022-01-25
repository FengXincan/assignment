'''
report:
I referred to this cite:
    https://www.jeddd.com/article/python-ngram-language-prediction.html
    that is am implement of n-gram on chinese news next word prediction.
we are different in:
    the data preprocessing,
    smoothing method,
    and also the entropy calculation.
'''


from collections import Counter
from math import log

# read train/test data into sentences
def read_sentences(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        for item in range(len(lines)):
            sentences.append('<s> ' + lines[item].rstrip('.,\n') + '</s>')
    return sentences

train_sentences = read_sentences('wiki-en-train.word')

# make word dictionary
def make_word_dic(sentences):
    word_dic = {}
    for item in sentences:
        splitted_sentence = item.split(' ')
        for i in range(len(splitted_sentence)):    
            word = splitted_sentence[i]
            word_dic[word] = word_dic.get(word, 0) + 1
    return word_dic

word_dic = make_word_dic(train_sentences)

'''
construct n-gram corpus:
read each sentence,
extract all n-gram lists, and all (n-1)-gram lists in each sentence,
count the number of all n-gram lists and all (n-1)-gram lists,
'''
def construct_ngram_corpus(sentences, n=3):
    ngram_list = [] # n grams
    prefix_list = [] # n-1 grams
    unigram_list = [] # n-2 grams
    for i, item in enumerate(sentences):
        ngram = list(zip(*[item.split()[i:] for i in range(n)]))
        prefix = list(zip(*[item.split()[i:] for i in range(n-1)]))
        unigram = list(zip(*[item.split()[i:] for i in range(n-2)]))
        ngram_list += ngram
        prefix_list += prefix
        unigram_list += unigram

    ngram_counter = Counter(ngram_list)
    prefix_counter = Counter(prefix_list)
    unigram_counter = Counter(unigram_list)
    return ngram_counter, prefix_counter, unigram_counter

ngram_counter, prefix_counter, unigram_counter = construct_ngram_corpus(train_sentences)

# calculate the probability of the sentence
def calculate_sentence_entropy(sentences, n=3):
    entropy_list = []
    probability = 1
    entropy = 0
    for i, item in enumerate(sentences):
        ngram = list(zip(*[item.split()[i:] for i in range(n)]))
        for piece in ngram:
            # add-1 smoothing
            # probability = ((ngram_counter[piece]+1) / (prefix_counter[(piece[0], piece[1])] + len(prefix_counter[(piece[0], piece[1])])))
            # Kneser-Ney Smoothing
            d = 0.75
            interpolation_weight = (d / unigram_counter[piece[1]]) * abs(prefix_counter[(piece[1], piece[2])])
            continue_probability = unigram_counter[piece[2]]
            probability = max((ngram_counter[piece] - d), 0) / prefix_counter[(piece[0], piece[1])] + interpolation_weight * continue_probability
            entropy += -log(probability)
            avg_entropy = entropy / len(item)
            entropy_list.append(avg_entropy)
    return entropy_list

test_sentences = read_sentences('wiki-en-test.word')
print(calculate_sentence_entropy(test_sentences))