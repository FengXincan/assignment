'''
summary:

'''


from collections import Counter

# read train data into sentences
def read_sentences(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        for item in range(len(lines)):
            sentences.append('<BOS> ' + lines[item].rstrip('.,\n') + '<EOS>')
    return sentences

sentences = read_sentences('wiki-en-train.word')

# make word dictionary
def make_word_dic(sentences):
    word_dic = {}
    for item in sentences:
        splitted_sentence = item.split(' ')
        for i in range(len(splitted_sentence)):    
            word = splitted_sentence[i]
            word_dic[word] = word_dic.get(word, 0) + 1
    return word_dic

word_dic = make_word_dic(sentences)

'''
construct n-gram corpus:
read each sentence,
extract all n-gram lists, and all (n-1)-gram lists in each sentence,
count the number of all n-gram lists and all (n-1)-gram lists,
'''
def construct_ngram_corpus(n, sentences):
    ngram_list = [] # n grams
    prefix_list = [] # n-1 grams
    for i, item in enumerate(sentences):
        ngram = list(zip(*[item.split()[i:] for i in range(n)]))
        prefix = list(zip(*[item.split()[i:] for i in range(n-1)]))
        ngram_list += ngram
        prefix_list += prefix

    ngram_counter = Counter(ngram_list)
    prefix_counter = Counter(prefix_list)
    return ngram_counter, prefix_counter

ngram_counter, prefix_counter = construct_ngram_corpus(3, sentences)

# hyperparameter of knears smoothing
d = 0.75
lam = 0.05
