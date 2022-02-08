# training part
# read training data
def read_sentences(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        sentences = []
        for item in range(len(lines)):
            sentences.append(lines[item].rstrip('.,\n'))
    return sentences

train_sentences = read_sentences('wiki-en-train.norm_pos')
# clean data, set the unknown token as 'UNK'
# do train


# testing part
# read test data
# clean data, replace the unkown token with 'UNK'
# do test