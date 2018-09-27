from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.losses import *
from data_reader import *
from PIL import Image
import numpy as np


def preprocessing_function(x):
    x *= 1 / 127.5
    x -= 1
    return x


def save_model(model, filepath):
    model.save(filepath)
    print('Save completed.')


class Predictor:
    def __init__(self, filepath, input_shape):
        self.filepath = filepath
        self.input_shape = input_shape
        self.width, self.height, self.n_dims = input_shape
        self.model = load_model(filepath)

    def predict(self, images):
        n_images = len(images)
        x = np.zeros([n_images] + list(self.input_shape))
        for i, image in enumerate(images):
            im = Image.open(image)
            im = im.resize((self.width, self.height))
            x[i, ...] = np.asarray(im)
        x = preprocessing_function(x)
        return self.model.predict(x, n_images)


class FoodClassifier:
    def __init__(self, path, batch_size, input_shape, classes, epoch, weight, logs):
        self.path = path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.classes = classes
        self.n_classes = len(classes)
        self.epoch = epoch
        self.weight = weight
        self.logs = logs
        # self.model = self.build_model()
        self.model = self.build_model_xception()

        self.model.compile(
            optimizer=Adam(),
            loss=categorical_crossentropy,
            metrics=['acc']
        )

        if os.path.isfile(weight):
            self.model.load_weights(weight)
            print('Load weight successfully.')
        else:
            print('No weight found.')

    def train(self):
        all_images = get_all_files(self.path)
        print('%s images found.' % len(all_images))
        generator = self.build_generator()
        self.model.fit_generator(
            generator=generator,
            steps_per_epoch=len(all_images) // self.batch_size,
            epochs=self.epoch,
            callbacks=[
                ModelCheckpoint('weight/{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5'),
                TensorBoard(self.logs)
            ]
        )

    def build_generator(self):
        generator = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=10,
            height_shift_range=10,
            preprocessing_function=preprocessing_function,
        )
        return generator.flow_from_directory(
            directory=self.path,
            target_size=self.input_shape[:2],
            classes=self.classes,
        )

    def build_model_xception(self):
        xception = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg',
            classes=self.n_classes,
        )

        x = Dense(self.n_classes, activation='softmax')(xception.output)

        model = Model(xception.input, x)

        model.summary()

        return model

    def build_model(self):
        x_input = Input(shape=self.input_shape)

        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x_input)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)

        x = Dense(self.n_classes, activation='softmax')(x)

        model = Model(x_input, x)

        model.summary()

        return model


WIDTH = 71
HEIGHT = 71
N_DIMS = 3
INPUT_SHAPE = (HEIGHT, WIDTH, N_DIMS)

PATH = 'data'
BATCH_SIZE = 64
CLASSES = os.listdir(PATH)
EPOCH = 10

PATH_WEIGHT = 'weight/'
PATH_MODEL = 'model/model.h5'
PATH_TEST = 'test'
PATH_TEST_IMAGES = [os.path.join(PATH_TEST, sub) for sub in os.listdir(PATH_TEST)]
PATH_LOGS = 'logs/baseline_xception'

if __name__ == '__main__':
    # Train
    classifier = FoodClassifier(PATH, BATCH_SIZE, INPUT_SHAPE, CLASSES, EPOCH, PATH_WEIGHT, PATH_LOGS)
    classifier.train()

    # # Dump Model
    # save_model(classifier.model, PATH_MODEL)

    # # Predict
    # predictor = Predictor(PATH_MODEL, INPUT_SHAPE)
    # images = PATH_TEST_IMAGES
    # predictions = predictor.predict(images)
    # for prediction, image in zip(predictions, images):
    #     index = int(np.argmax(prediction))
    #     prob = prediction[index]
    #     result = CLASSES[index]
    #     print('%s: %s (%.1f%%)' % (image, result, prob * 100))
