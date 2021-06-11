"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import argparse
import tensorflow as tf
from vgg_model import VGGModel
from your_model import YourModel
import hyperparameters as hp
from preprocess import Datasets
from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '2'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (2).''')
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/../data/',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default=os.getcwd() + '/vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 2).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. In
        the case of task 2, passing a checkpoint path will disable
        the loading of VGG weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(datasets)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    datasets = Datasets(ARGS.data, ARGS.task)

    if ARGS.task == '1':
        model = YourModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "./your_model_checkpoints/"
        model.summary()
    else:
        model = VGGModel()
        checkpoint_path = "./vgg_model_checkpoints/"
        model(tf.keras.Input(shape=(224, 224, 3)))
        model.summary()

        # Don't load pretrained vgg if loading checkpoint
        if ARGS.load_checkpoint is None:
            model.load_weights(ARGS.load_vgg, by_name=True)

    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path)

# Make arguments global
ARGS = parse_args()

main()
