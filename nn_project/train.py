import datetime
import math

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from nn_project.data import get_data_generator, get_vocabs, MAX_ROWS_TRAIN
from nn_project.model import EncoderDecoder


def train(
        data_limit=None,
        data_offset=None,
        validation_split=0.2,
        vocab_size=None,
        padding='post',
        cell_type='lstm',
        input_embedding_size=200,
        context_vector_size=2000,
        output_embedding_size=200,
        learning_rate=0.001,
        batch_size=32,
        epochs=1000,
        early_stopping=5,
        save_models=True,
        save_logs=True,
        queue_size=10,
):
    data_offset = data_offset if data_offset is not None else 0
    data_limit = data_limit if data_limit is not None else MAX_ROWS_TRAIN
    data_limit = min(data_limit, MAX_ROWS_TRAIN - data_offset)
    vocab_en, vocab_cs, length_en, length_cs = get_vocabs(
        kind='train',
        limit=data_limit,
        offset=data_offset,
        vocab_size=vocab_size,
    )
    training_offset = data_offset
    training_limit = int(data_limit * (1.0 - validation_split))
    validation_offset = training_offset + training_limit
    validation_limit = data_limit - training_limit
    training_data = get_data_generator(
        kind='train',
        batch_size=batch_size,
        limit=training_limit,
        offset=training_offset,
        vocab_en=vocab_en,
        vocab_cs=vocab_cs,
        length_en=length_en,
        length_cs=length_cs,
        padding=padding,
    )
    validation_data = get_data_generator(
        kind='train',
        batch_size=batch_size,
        limit=validation_limit,
        offset=validation_offset,
        vocab_en=vocab_en,
        vocab_cs=vocab_cs,
        length_en=length_en,
        length_cs=length_cs,
        padding=padding,
    )
    model = EncoderDecoder(
        input_length=length_en,
        input_vocab_size=len(vocab_en),
        input_embedding_size=input_embedding_size,
        context_vector_size=context_vector_size,
        output_length=length_cs,
        output_vocab_size=len(vocab_cs),
        output_embedding_size=output_embedding_size,
        cell_type=cell_type,
        enable_masking=True,
    )
    model.compile(
        optimizer=Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=[SparseCategoricalAccuracy(name='accuracy')],
    )
    steps_per_epoch = math.ceil(training_limit / batch_size)
    validation_steps = math.ceil(validation_limit / batch_size)
    callback_list = get_callbacks(
        early_stopping=early_stopping,
        save_models=save_models,
        save_logs=save_logs,
    )
    history = model.fit_generator(
        generator=training_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callback_list,
        validation_data=validation_data,
        validation_steps=validation_steps,
        max_queue_size=queue_size,
    )
    model.summary()
    return history


def get_callbacks(early_stopping, save_models, save_logs):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callback_list = []
    if early_stopping is not None and early_stopping is not False:
        callback = callbacks.EarlyStopping(patience=early_stopping)
        callback_list.append(callback)
    if save_models:
        latest_callback = callbacks.ModelCheckpoint(
            filepath=f'../models/{timestamp}-latest',
        )
        best_callback = callbacks.ModelCheckpoint(
            filepath=f'../models/{timestamp}-best',
            save_best_only=True,
        )
        callback_list += [latest_callback, best_callback]
    if save_logs:
        callback = callbacks.TensorBoard(
            log_dir=f'../logs/{timestamp}',
        )
        callback_list.append(callback)
    return callback_list


if __name__ == '__main__':
    train()
