import numpy as np
import cv2
import random
from glob import glob

from keras.utils import to_categorical

class InputGenerator(object):
    def  __init__(self, datapath, input_size):
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

    def generator(self, batch_size=32):

        data_copy = self.data.copy()
        random.shuffle(data_copy)

        while True:
            if len(data_copy) < batch_size:
                data_copy = self.data.copy()
                random.shuffle(data_copy)
            X = []
            y = {"fc_speed": [], "fc_direction": []}
            for _ in range(batch_size):
                input_path, output = data_copy[0]
                # Input formating
                input = cv2.imread(input_path)
                """Data augmentation there"""

                # Normalization
                size = (self.input_size[1], self.input_size[0])
                input = cv2.resize(input, size)
                input = np.float32(input)
                input /= 255.

                # Output formating
                speed, direction = output
                speed, direction = to_categorical(speed, 2), to_categorical(direction, 5)

                X.append(input)
                y['fc_speed'].append(speed)
                y['fc_direction'].append(direction)
                del data_copy[0]

            X = np.array(X)
            y['fc_speed'] = np.array(y['fc_speed'])
            y['fc_direction'] = np.array(y['fc_direction'])
            yield X, y
