from const import *
import numpy as np
import sys
import time

import Adafruit_PCA9685
from keras.models import load_model
from collections import deque

from pivideostream import PiVideoStream

memory = deque(maxlen=6)

#Load model
model = load_model(sys.argv[1])
print("Model loaded")

# Init engines
speed = SPEED_FAST
direction = DIR_C
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(50)

# created a *threaded *video stream, allow the camera sensor to warmup
vs = PiVideoStream().start()
time.sleep(2.0)

from PIL import Image
frame = vs.read()
img = Image.fromarray(frame)
img.save("test.png")


# Starting loop
print("Ready ! press CTRL+C to START/STOP :")
try:
    while True:
        pass
except KeyboardInterrupt:
    pass

# Handle START/STOP event
try:
    head = H_UP
    # loop over some frames...this time using the threaded stream
    while True:
            # grab the frame from the threaded video stream
            frame = vs.read()
            image = np.array([frame]) / 255.0
            ##  # Model prediction
            preds_raw = model.predict(image)
            memory.append(preds_raw[1])
            preds_raw[1] = np.asarray(memory).sum(axis=0)
            preds = [np.argmax(pred, axis=1) for pred in preds_raw]
            ##  # Action
            if preds[1] == 0:
                if speed == SPEED_NORMAL:
                    head = H_DOWN
                speed = SPEED_NORMAL
                direction = DIR_L_M
            elif preds[1] == 1:
                speed = SPEED_NORMAL
                direction = DIR_L
            elif preds[1] == 2:
                if preds[0] == 1:
                    if speed == SPEED_FAST:
                        head = H_UP
                    speed = SPEED_FAST
                else:
                    speed = SPEED_NORMAL
                direction = DIR_C
            elif preds[1] == 3:
                speed = SPEED_NORMAL
                direction = DIR_R
            elif preds[1] == 4:
                if speed == SPEED_NORMAL:
                    head = H_DOWN
                speed = SPEED_NORMAL
                direction = DIR_R_M
            ##  # Apply values to engines
            pwm.set_pwm(0, 0, direction)
            pwm.set_pwm(1, 0, speed)
            ## # Move Head
            pwm.set_pwm(2, 0, head)

except:
    pass

# Stop the machine
pwm.set_pwm(0, 0, 0)
pwm.set_pwm(1, 0, 0)
pwm.set_pwm(2, 0, 0)
vs.stop()
print("Stop")
