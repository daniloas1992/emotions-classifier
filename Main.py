import os

# Enable GPU:
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime
import time
import glob
import cv2
import numpy as np
from enum import Enum
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow_core.python.keras.utils import np_utils
from tensorflow.python.client import device_lib

#Ignore:
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfiProto(gpu_options=gpu_options))

#Disable GPU:
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 0}))

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

            image = cv2.imread(url_image, cv2.IMREAD_GRAYSCALE)

            image_resized = cv2.resize(image, dimension_resized)
            dataset.append(Face(image_resized, emotion_face, label))

        except Exception as e:
            pass


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

    y = np.reshape(y, (len(y),1))
    return x, y


def train_data(features, labels):

    features = features / 255.0
    y_labels = np_utils.to_categorical(labels)

    dense_layers = [0, 1, 2, 3, 4, 5]
    sizes_layers = [32, 64, 128, 256]
    conv_layers = [1, 2, 3, 4]

    for dense in dense_layers:
        for size in sizes_layers:
            for conv in conv_layers:

                name_model = 'Training_Model_{}_Dense_{}_Size_{}_Conv_{}'.format(dense,size,conv,int(time.time()))

                tensorboard = TensorBoard(log_dir='logs\\{}'.format(name_model))

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

    '''model = Sequential()

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

    model.fit(features, y_labels, batch_size=32, epochs=40, validation_split=0.1, callbacks=[tensorboard])'''


def main():

    dataset = []
    dir_image_files = "C:/Users/Danilo/Desktop/DATASETS/KDEF/"
    width = 50
    height = 67
    features = None
    labels = None

    menu = Menu()
    option = True

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
            else:
                print('Dataset not found! Create a dataset before load it!')

        elif option == '3':
            if features is None or labels is None:
                print('You need load a dataset before use it!')
            else:
                print('\nTraining dataset...')
                train_data(features, labels)

        elif option == '4':
            break

        else:
            print('\nInvalid entrance! Select another option!')


if __name__ == "__main__":
    main()
