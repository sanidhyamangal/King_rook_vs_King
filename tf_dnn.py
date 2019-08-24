# import tf for classification
import tensorflow as tf
from load_data import LoadData

# make a data loader obj
loadData = LoadData("./kr-vs-k.csv")

# retrive the data dict
data = loadData.load_processed_data()

# a input function
def input_fn_generator(data):
    def input_fn():
        # load the features
        features = data["features"].values
        labels = data["labels"].values

        # convert the data into tensors
        features = tf.convert_to_tensor(value=features, dtype=tf.float32)
        labels = tf.convert_to_tensor(value=labels, dtype=tf.int32)

        # make a tensor dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        # make a batch of data
        dataset = dataset.shuffle(100).batch(64)

        # make a oneshot iterator
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

        batch_features, batch_labels = iterator.get_next()

        return {"attributes": batch_features}, batch_labels

    return input_fn


# a function to handdle all the new and better functions
def dnn_model(features, labels, mode):
    # make a first layer for the input
    inputs = features["attributes"]

    # make a fully connected layer
    # dense layer 1
    dense1 = tf.compat.v1.layers.dense(inputs, units=64)

    # make a drop out layer
    dropout1 = tf.compat.v1.layers.dropout(
        dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # make a dense connected layer
    dense2 = tf.compat.v1.layers.dense(dropout1, units=32)

    # make a new drop out layer
    dropout2 = tf.compat.v1.layers.dropout(
        dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # make a final dense layer
    logits = tf.compat.v1.layers.dense(dropout2, units=18)

    # make a predictions
    predictions = {
        "class": tf.argmax(logits, axis=1),
        "probablity": tf.nn.softmax(logits=logits, name="softmax_tensor"),
    }

    # return the predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # compute the losses for model training
    losses = tf.compat.v1.losses.softmax_cross_entropy(labels, logits=logits)

    # make a training ops
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=losses, global_step=tf.compat.v1.train.get_global_step()
        )

        return tf.estimator.EstimatorSpec(mode, loss=losses, train_op=train_op)

    # compute the eval task
    eval_ops = {
        "accuracy": tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions["class"]
        )
    }

    # if mode is eval mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, eval_metric_ops=eval_ops)


# make a dense classifie
dnn = tf.estimator.Estimator(model_fn=dnn_model, model_dir="./tf_model")

# train dnn estimator
dnn.train(input_fn=input_fn_generator(data), steps=100, hooks=None)
