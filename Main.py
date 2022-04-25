import os
import sys
# Enable GPU:
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time
import glob
#import cv2
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from tensorflow.keras.utils import to_categorical

from tensorflow.python.client import device_lib
np.random.seed(42)

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

gpus = tf.config.experimental.list_physical_devices('GPU')

config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 4))])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    print('ATENTION: GPU device not found. CPU will be used!')
else:
    print('Found GPU at: {}'.format(device_name))


class Menu:
    menu_option = {'1': 'Generate dataset',
                   '2': 'Load dataset',
                   '3': 'Training Classifier',
                   '4': 'Exit'}

    def show_menu(self):
        print('\n')
        for key, value in self.menu_option.items():
            print('%s -> %s' % (key, value))

    def get_menu_option(self):

        chosen = input('\nSelect an option -> ')

        try:
            if chosen in self.menu_option:
                return chosen
            else:
                return -1

        except KeyError:
            return -1


class Emotion(Enum):

    AFRAID = 'AF'
    ANGRY = 'AN'
    DISGUSTED = 'DI'
    HAPPY = 'HA'
    NEUTRAL = 'NE'
    SAD = 'SA'
    SURPRISED = 'SU'


class Face:

    def __init__(self, image_array, emotion, label):
        self.image_array = image_array
        self.emotion = emotion
        self.label = label

    def get_image_array(self):
        return self.image_array

    def get_emotion(self):
        return self.emotion

    def get_label(self):
        return self.label

    @staticmethod
    def count_emotions(faces):
        emotions = {Emotion('AF').name: 0,
                    Emotion('AN').name: 0,
                    Emotion('DI').name: 0,
                    Emotion('HA').name: 0,
                    Emotion('NE').name: 0,
                    Emotion('SA').name: 0,
                    Emotion('SU').name: 0}

        for i in range(len(faces)):
            emotions[faces[i].get_emotion()] += 1

        print('Emotion distributed Dataset KDEF:\n')
        for key, value in emotions.items():
            print('%s: %s' % (key, value))


def get_label(emotion):

    labels = {'AFRAID': 0,
              'ANGRY': 1,
              'DISGUSTED': 2,
              'HAPPY': 3,
              'NEUTRAL': 4,
              'SAD': 5,
              'SURPRISED': 6}

    return labels.get(emotion,'')


def create_dataset(dataset, dir_files, dimension_resized):

    for dir in glob.iglob(dir_files + '**/*', recursive=False):

        try:
            url_image = dir.replace('\\', '/')
            emotion_face = Emotion((str(dir).split('\\')[-1])[4:6]).name

            label = get_label(str(emotion_face))

            #image = cv2.imread(url_image, cv2.IMREAD_GRAYSCALE)

            #image_resized = cv2.resize(image, dimension_resized)
            #dataset.append(Face(image_resized, emotion_face, label))

        except Exception as err:
            print('Error creating dataset: ' + err)


def delete_dataset():
    if os.path.exists("features.npy"):
        os.remove("features.npy")

    if os.path.exists("labels.npy"):
        os.remove("labels.npy")


def get_dataset_data(dataset, dimension):
    x = []
    y = []

    for face in dataset:
        image_array = face.get_image_array()
        x.append(np.reshape(image_array, dimension))
        y.append(face.get_label())

    y = np.reshape(y, (len(y), 1))
    return x, y


def train_data(features, labels):

    features = features / 255.0
    y_labels = to_categorical(labels)

    dense_layers = [0, 1, 2, 3, 4, 5]
    sizes_layers = [32, 64, 128, 256]
    conv_layers = [1, 2, 3, 4]

    for dense in dense_layers:
        for size in sizes_layers:
            for conv in conv_layers:

                name_model = 'Training_Model_{}_Dense_{}_Size_{}_Conv_{}'.format(dense, size, conv, int(time.time()))

                tensorboard = TensorBoard(log_dir='logs/{}'.format(name_model))

                model = Sequential()

                model.add(Conv2D(size, (3, 3), input_shape=features.shape[1:]))
                model.add(Activation("relu"))
                model.add(MaxPool2D(pool_size=(2, 2)))

                for layer in range(conv-1):
                    model.add(Conv2D(size, (3, 3)))
                    model.add(Activation("relu"))
                    model.add(MaxPool2D(pool_size=(2, 2)))

                model.add(Flatten())

                for layer in range(dense):
                    model.add(Dense(size))
                    model.add(Activation('relu'))

                model.add(Dense(7))
                model.add(Activation('softmax'))

                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                print(name_model)
                print(model.summary())
                model.fit(features, y_labels, batch_size=32, epochs=25, validation_split=0.1, callbacks=[tensorboard])


def train_data_dev(features, labels):

    features = features / 255.0

    fold = 0
    y_test_values = []
    '''result_predict = []

    fscore = []
    precision = []
    recall = []
    roc = []'''

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train, test in skf.split(features, labels):

        fold += 1

        y_labels = to_categorical(labels)

        x_train = features[train]
        y_train = y_labels[train]
        x_test = features[test]
        y_test = y_labels[test]

        print('\n### TRAINNING FOLD {} \nX: {} \nY: {} '.format(str(fold), str(x_train.shape), str(y_train.shape)))

        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=features.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(7))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        tensorboard = TensorBoard(log_dir='logsDev/{}_{}'.format('Training_Model_Dev', str(fold)))
        model.fit(x_train, y_train, batch_size=512, epochs=50, validation_split=0.1, callbacks=[tensorboard])

        y_pred = model.predict(x_test)

        y_test_values.append(y_test)
        #result_predict.append(y_pred)

        confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

        plt.figure(figsize=(12, 12))
        plt.title('TRAINNING FOLD {} - X: {} - Y: {} '.format(str(fold), str(x_train.shape), str(y_train.shape)))
        columns = [f'Pred {label}' for label in ('Afraid', 'Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad', 'Surprised')]
        indices = [f'Actual {label}' for label in ('Afraid', 'Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad', 'Surprised')]

        table = pd.DataFrame(confusion_matrix, columns=columns, index=indices)
        sns.heatmap(table, annot=True, fmt='d')
        plt.savefig('images/image_{}.png'.format(str(datetime.now())))
        plt.close()

        '''y_test = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)

        fscore.append(metrics.f1_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        roc.append(metrics.roc_auc_score(y_test, y_pred))

        sys.exit(0)
        #score = np.sqrt(metrics.accuracy_score(y_test, pred))
        #print('ACCURACY SCORE: {}'.format(score))'''

    #return {'test_f1': fscore, 'test_precision': precision, 'test_recall': recall, 'test_roc_auc': roc}


def mlp_classifier(features, labels):

    features = features / 255.0
    y_labels = to_categorical(labels)
    mlp = MLPClassifier(hidden_layer_sizes=(7,), max_iter=25, tol=0.00001, solver='adam', activation='relu')

    kf = KFold(3, shuffle=True, random_state=42)

    fold = 0
    for train, test in kf.split(features, y_labels):
        fold += 1

        x_train = features[train]
        y_train = y_labels[train]
        x_test = features[test]
        y_test = y_labels[test]

        print('\n### TRAINNING FOLD {} \nX: {} \nY: {} '.format(str(fold), str(x_train.shape), str(y_train.shape)))

        mlp.fit(x_train, y_train)

        pred = mlp.predict(x_test)

        confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
        print(confusion_matrix)


'''def show_result(result):
    print('Test     ROC: %f' % (reduce(lambda x, y: x + y, result['score']['test_roc_auc']) / 10))
    print('Test      F1: %f' % (reduce(lambda x, y: x + y, result['score']['test_f1']) / 10))
    print('Test      F1: ' + str(result['score']['test_f1']))
    print('Test      Precision: ' + str(result['score']['test_precision']))
    print('Test      Recall: ' + str(result['score']['test_recall']))
    print('Best      F1: ' + str(result['score']['best_f1']))
    print('Best      F1: %f' % (reduce(lambda x, y: x + y, result['score']['best_f1']) / 10))'''


def main():

    dataset = []
    dir_image_files = "dataset/KDEF/"
    width = 50
    height = 67
    features = None
    labels = None

    menu = Menu()
    option = True
    dev = True

    while option:

        menu.show_menu()
        option = menu.get_menu_option()

        if option == '1':
            print('\nCreating dataset...')
            delete_dataset()
            start = datetime.now()
            create_dataset(dataset, dir_image_files, (width, height))
            print('\nTotal time creating dataset: {} seconds.\n'.format((datetime.now() - start).total_seconds()))
            Face.count_emotions(dataset)
            features, labels = get_dataset_data(dataset, (width, height, 1))
            np.save('features.npy', features)
            np.save('labels.npy', labels)

        elif option == '2':
            if os.path.exists("features.npy") and os.path.exists("labels.npy"):
                print('\nLoading dataset...')
                features = np.load('features.npy')
                labels = np.load('labels.npy')
                print('\nDataset loaded...\nX: {} \nY: {} '.format(str(features.shape), str(labels.shape)))
            else:
                print('Dataset not found! Create a dataset before load it!')

        elif option == '3':
            if features is None or labels is None:
                print('You need load a dataset before use it!')
            else:

                if dev:
                    print('\nMode DEV: Training dataset...')
                    #mlp_classifier(features, labels)
                    result = train_data_dev(features, labels)
                    #show_result(result)
                else:
                    print('\nTraining dataset...')
                    train_data(features, labels)

        elif option == '4':
            break

        else:
            print('\nInvalid entrance! Select another option!')


if __name__ == "__main__":
    main()
