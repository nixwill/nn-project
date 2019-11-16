import collections
import itertools


def get_data(kind='train', limit=None, vocab_size=None):
    data_en = open(f'../data/raw/{kind}.en.txt', 'r')
    data_cs = open(f'../data/raw/{kind}.cs.txt', 'r')
    with data_en, data_cs:
        lines_en = get_lines(data=data_en, limit=limit)
        lines_cs = get_lines(data=data_cs, limit=limit)
    vocab_en = make_vocab(lines=lines_en, size=vocab_size)
    vocab_cs = make_vocab(lines=lines_cs, size=vocab_size)
    samples = get_samples(
        lines_en=lines_en,
        lines_cs=lines_cs,
        vocab_en=vocab_en,
        vocab_cs=vocab_cs,
    )
    return samples, vocab_en, vocab_cs


def get_lines(data, limit=None):
    lines = [line.strip() for line in itertools.islice(data, limit)]
    return lines


def make_vocab(lines, size=None):
    words = [word for line in lines for word in line.split()]
    counter = collections.Counter(words)
    vocab = {}
    for index, (word, _) in enumerate(counter.most_common(size), start=1):
        vocab[word] = index
    return vocab


def get_samples(lines_en, lines_cs, vocab_en, vocab_cs):
    for line_en, line_cs in zip(lines_en, lines_cs):
        sample_en = list(encode_sample(line=line_en, vocab=vocab_en))
        sample_cs = list(encode_sample(line=line_cs, vocab=vocab_cs))
        yield sample_en, sample_cs


def encode_sample(line, vocab):
    tokens = line.split()
    for token in tokens:
        token_id = vocab.get(token, 0)
        yield token_id
