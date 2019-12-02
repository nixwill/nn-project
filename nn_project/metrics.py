import tensorflow as tf


def accuracy(y_true, y_pred):
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=y_true.dtype)
    mask = tf.where(tf.logical_or(y_true != 0, y_pred != 0), 1.0, 0.0)
    correct = mask * tf.where(y_true == y_pred, 1.0, 0.0)
    return tf.reduce_sum(correct, axis=-1) / tf.reduce_sum(mask, axis=-1)


def word_error_rate(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1, output_type='int64')
    y_true = tf.cast(y_true, dtype='int64')
    y_pred_indices = tf.where(y_pred != 0)
    y_pred = tf.SparseTensor(
        indices=y_pred_indices,
        values=tf.gather_nd(y_pred, y_pred_indices),
        dense_shape=tf.shape(y_pred, out_type='int64'),
    )
    y_true_indices = tf.where(y_true != 0)
    y_true = tf.SparseTensor(
        indices=y_true_indices,
        values=tf.gather_nd(y_true, y_true_indices),
        dense_shape=tf.shape(y_true, out_type='int64'),
    )
    return tf.edit_distance(y_pred, y_true)
