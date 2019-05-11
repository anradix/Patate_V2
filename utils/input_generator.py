import numpy as np
import cv2
import random
from glob import glob

from keras.utils import to_categorical

class InputGenerator(object):
    def  __init__(self,
                datapath,
                input_size,
                augmentation = {
                    "dropout_coarse": 0.33,
                    "gray_scale": 0.33,
                    "flip_vertical": 0.33,
                    "noise_gaussian": 0.33,
                    "blur_gaussian": 0.33,
                }
            ):
        dir_list = glob(datapath + '/*.jpg')

        data = []
        # Create shapes
        for path in dir_list:
            filename = path.split('/')[-1]
            speed, dir, _  = filename.split('_')
            data.append([path, (speed, dir)])

        self.data = data
        self.size = len(data)
        self.input_size = input_size
        self.debug = False

        self.__dict__.update(augmentation)

    def augGrayScale(self, input):
        # ton code ici
        input = np.dot(input[...,:3], [0.299, 0.587, 0.114])
        input = np.reshape(input, (input.shape[0], input.shape[1], 1))
        input = np.concatenate((input, input, input), axis=2)
        return input

    def augGaussianBlur(self, input):
        kernel = (5,5)
        input = cv2.GaussianBlur(input, kernel, 0)
        return input

    def augGaussianNoise(self, input):
        mean, var = 0, 10
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, input.shape[:2]) #  np.zeros((224, 224), np.float32)
        noisy_image = np.zeros(input.shape, np.float32)
        if len(input.shape) == 2:
            noisy_image = img + gaussian
        else:
            noisy_image[:, :, 0] = input[:, :, 0] + gaussian
            noisy_image[:, :, 1] = input[:, :, 1] + gaussian
            noisy_image[:, :, 2] = input[:, :, 2] + gaussian
        return noisy_image

    def augFlipVertical(self, input, output):
        pass
        # ton code ici
        return input, output

    def augCoarseDropout(self, input):
        scale, shape = random.randint(5, 30), input.shape[:2]
        shape = (np.int32(shape[0] / scale), np.int32(shape[1] / scale))
        mask = np.random.randint(2, size=shape).reshape([shape[0], shape[1], 1]) | np.random.randint(2, size=shape).reshape([shape[0], shape[1], 1])
        mask = np.concatenate((mask, mask, mask), axis=2)
        input = input * cv2.resize(np.uint8(mask), (input.shape[1], input.shape[0]), interpolation=cv2.INTER_NEAREST)
        return input

    def augmentation(self, input, output):
        if random.random() < self.blur_gaussian:
            input = self.augGaussianBlur(input)
        if random.random() < self.gray_scale:
            input = self.augGrayScale(input)
        if random.random() < self.flip_vertical:
            input, output = self.augFlipVertical(input, output)
        if random.random() < self.noise_gaussian:
            input = self.augGaussianNoise(input)
        if random.random() < self.dropout_coarse:
            input = self.augCoarseDropout(input)

        return input, output

    def generator(self, batch_size=32):

        data_copy = self.data.copy()
        random.shuffle(data_copy)

        while True:
            if len(data_copy) < batch_size:
                data_copy = self.data.copy()
                random.shuffle(data_copy)
            X = []
            y = {"speed": [], "direction": []}
            for _ in range(batch_size):
                input_path, output = data_copy[0]
                # Input formating
                input = cv2.imread(input_path)

                """ Data augmentation there
                """
                input, output = self.augmentation(input, output)

                # Normalization
                size = (self.input_size[1], self.input_size[0])
                input = cv2.resize(input, size)
                input = np.float32(input)
                input /= 255.

                # Output formating
                speed, direction = output
                speed, direction = to_categorical(speed, 2), to_categorical(direction, 5)

                X.append(input)
                y['speed'].append(speed)
                y['direction'].append(direction)
                del data_copy[0]

            X = np.array(X)
            y['speed'] = np.array(y['speed'])
            y['direction'] = np.array(y['direction'])
            yield X, y


if __name__ == "__main__":

    aug = {
        "gray_scale": 0.,
        "gray_scale": 0.,
        "flip_vertical": 0.,
        "noise_gaussian": 0.,
        "blur_gaussian": 0.,
    }
    gen = InputGenerator("./datas/Train", (96, 160, 3))
    for X, y in gen.generator(batch_size=1):
        input = X[0] * 255
        input = cv2.resize(input, (1280, 768), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('image',np.uint8(input))
        cv2.waitKey(0)
