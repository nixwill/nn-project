import math

from tensorflow.keras.models import load_model

from nn_project.data import MAX_ROWS_TRAIN, get_vocabs, get_data_generator
from nn_project.utils import get_project_file


def test(
        model_id,
        train_limit=None,
        train_offset=None,
        test_limit=None,
        test_offset=None,
        vocab_size=None,
        padding='post',
        batch_size=128,
        queue_size=10,
):
    train_offset = train_offset if train_offset is not None else 0
    train_limit = train_limit if train_limit is not None else MAX_ROWS_TRAIN
    train_limit = min(train_limit, MAX_ROWS_TRAIN - train_offset)
    vocab_en, vocab_cs, length_en, length_cs = get_vocabs(
        kind='train',
        limit=train_limit,
        offset=train_offset,
        vocab_size=vocab_size,
    )
    test_data = get_data_generator(
        kind='test',
        batch_size=batch_size,
        limit=test_limit,
        offset=test_offset,
        vocab_en=vocab_en,
        vocab_cs=vocab_cs,
        length_en=length_en,
        length_cs=length_cs,
        padding=padding,
    )
    model = load_model(get_project_file('models', model_id))
    metrics = model.fit_generator(
        generator=test_data,
        steps=math.ceil(test_limit / batch_size),
        max_queue_size=queue_size,
    )
    return metrics
