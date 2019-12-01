import collections
import itertools


def get_data(
        kind='train',
        limit=None,
        offset=None,
        vocab_size=None,
        encoded=False,
):
    data_en = open(f'../data/raw/{kind}.en.txt', 'r')
    data_cs = open(f'../data/raw/{kind}.cs.txt', 'r')
    with data_en, data_cs:
        lines_en = get_lines(data=data_en, limit=limit, offset=offset)
        lines_cs = get_lines(data=data_cs, limit=limit, offset=offset)
    vocab_en = make_vocab(lines=lines_en, size=vocab_size)
    vocab_cs = make_vocab(lines=lines_cs, size=vocab_size)
    if encoded:
        samples_en, samples_cs = get_samples(
            lines_en=lines_en,
            lines_cs=lines_cs,
            vocab_en=vocab_en,
            vocab_cs=vocab_cs,
        )
    else:
        samples_en, samples_cs = get_samples(
            lines_en=lines_en,
            lines_cs=lines_cs,
        )
    return samples_en, samples_cs, vocab_en, vocab_cs


def get_lines(data, limit=None, offset=None):
    start = 0 if offset is None else offset
    stop = None if limit is None else start + limit
    lines = [line.strip() for line in itertools.islice(data, start, stop)]
    return lines


def make_vocab(lines, size=None):
    words = [word for line in lines for word in line.split()]
    counter = collections.Counter(words)
    vocab = {'<?>': 0}
    for index, (word, _) in enumerate(counter.most_common(size), start=1):
        vocab[word] = index
    return vocab


def get_samples(lines_en, lines_cs, vocab_en=None, vocab_cs=None):
    samples_en = [
        list(encode_sample(line=line, vocab=vocab_en))
        for line in lines_en
    ]
    samples_cs = [
        list(encode_sample(line=line, vocab=vocab_cs))
        for line in lines_cs
    ]
    return samples_en, samples_cs


def encode_sample(line, vocab=None):
    tokens = line.split()
    for token in tokens:
        if vocab is None:
            yield token
        else:
            yield vocab.get(token, 0)
