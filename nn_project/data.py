import collections
import itertools

from tensorflow.keras.preprocessing.sequence import pad_sequences

from nn_project.utils import get_project_file


def get_data_generator(
        kind='train',
        batch_size=1,
        limit=None,
        offset=None,
        vocab_size=None,
        vocab_en=None,
        vocab_cs=None,
        length_en=None,
        length_cs=None,
        padding='post',
):
    if None in (vocab_en, vocab_cs, length_en, length_cs):
        vocab_en, vocab_cs, length_en, length_cs = get_vocabs(
            kind=kind,
            limit=limit,
            offset=offset,
            vocab_size=vocab_size,
        )
    data_generator = get_epochs(
        kind=kind,
        batch_size=batch_size,
        limit=limit,
        offset=offset,
        vocab_en=vocab_en,
        vocab_cs=vocab_cs,
        length_en=length_en,
        length_cs=length_cs,
        padding=padding,
    )
    return data_generator


def get_epochs(
        kind,
        batch_size,
        limit,
        offset,
        vocab_en,
        vocab_cs,
        length_en,
        length_cs,
        padding,
):
    while True:
        data_en, data_cs = get_files(kind=kind)
        samples_en = get_samples(
            data=data_en,
            limit=limit,
            offset=offset,
            vocab=vocab_en,
        )
        samples_cs = get_samples(
            data=data_cs,
            limit=limit,
            offset=offset,
            vocab=vocab_cs,
        )
        yield from get_batches(
            samples_en=samples_en,
            samples_cs=samples_cs,
            length_en=length_en,
            length_cs=length_cs,
            batch_size=batch_size,
            padding=padding,
        )
        data_en.close()
        data_cs.close()


def get_batches(
        samples_en,
        samples_cs,
        length_en,
        length_cs,
        batch_size,
        padding,
):
    while True:
        batch_en = list(itertools.islice(samples_en, batch_size))
        batch_cs = list(itertools.islice(samples_cs, batch_size))
        if not batch_en or not batch_cs:
            break
        batch_en = pad_sequences(
            sequences=batch_en,
            maxlen=length_en,
            padding=padding,
        )
        batch_cs = pad_sequences(
            sequences=batch_cs,
            maxlen=length_cs,
            padding=padding,
        )
        yield batch_en, batch_cs


def get_data(
        kind='train',
        limit=None,
        offset=None,
        vocab_en=None,
        vocab_cs=None,
        vocab_size=None,
        encoded=False,
):
    if vocab_en is None or vocab_cs is None:
        vocab_en, vocab_cs, length_en, length_cs = get_vocabs(
            kind=kind,
            limit=limit,
            offset=offset,
            vocab_size=vocab_size,
        )
    data_en, data_cs = get_files(kind=kind)
    with data_en, data_cs:
        if encoded:
            samples_en = get_samples(
                data=data_en,
                limit=limit,
                offset=offset,
                vocab=vocab_en,
            )
            samples_cs = get_samples(
                data=data_cs,
                limit=limit,
                offset=offset,
                vocab=vocab_cs,
            )
        else:
            samples_en = get_samples(
                data=data_en,
                limit=limit,
                offset=offset,
            )
            samples_cs = get_samples(
                data=data_cs,
                limit=limit,
                offset=offset,
            )
        return list(samples_en), list(samples_cs), vocab_en, vocab_cs


def get_vocabs(kind, limit=None, offset=None, vocab_size=None):
    data_en, data_cs = get_files(kind=kind)
    with data_en, data_cs:
        lines_en = get_lines(data=data_en, limit=limit, offset=offset)
        lines_cs = get_lines(data=data_cs, limit=limit, offset=offset)
        vocab_en, length_en = make_vocab(lines=lines_en, size=vocab_size)
        vocab_cs, length_cs = make_vocab(lines=lines_cs, size=vocab_size)
    return vocab_en, vocab_cs, length_en, length_cs


def make_vocab(lines, size=None):
    max_length = 0
    counter = collections.Counter()
    for line in lines:
        words = line.split()
        counter.update(words)
        max_length = max(len(words), max_length)
    vocab = {'<?>': 0}
    for index, (word, _) in enumerate(counter.most_common(size), start=1):
        vocab[word] = index
    return vocab, max_length


def get_samples(data, limit=None, offset=None, vocab=None):
    lines = get_lines(data=data, limit=limit, offset=offset)
    for line in lines:
        yield list(encode_sample(line=line, vocab=vocab))


def encode_sample(line, vocab=None):
    tokens = line.split()
    for token in tokens:
        if vocab is None:
            yield token
        else:
            yield vocab.get(token, 0)


def get_files(kind):
    data_en = open(get_project_file('data', 'raw', f'{kind}.en.txt'), 'r')
    data_cs = open(get_project_file('data', 'raw', f'{kind}.cs.txt'), 'r')
    return data_en, data_cs


def get_lines(data, limit=None, offset=None):
    start = 0 if offset is None else offset
    stop = None if limit is None else start + limit
    return itertools.islice(data, start, stop)


MAX_ROWS_TRAIN = 15794564
MAX_ROWS_TEST = 2656
