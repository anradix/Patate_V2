# Import the PCA9685 module.
import Adafruit_PCA9685

# import xbox driver
import xbox

import time
from picamera import PiCamera
from picamera.array import PiRGBArray
from PIL import Image
import sys

from const import *

#############################################

class Controler(object):
    def __init__(self):
        self.snap = False
        if len(sys.argv) == 2:
            self.delay = float(sys.argv[1])
            self.snap = True
        else:
            print("No Snaps, specify a delay value (float) to activate")

        self.label = [-1, 2]

        #set controls
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(50)

        # Init speed
        self.speed = SPEED_NORMAL
        # Init direction
        self.direction = DIR_C
        
        # Setup Camera
        self.camera = PiCamera()
        self.camera.resolution = IM_SIZE
        self.camera.framerate = 60
        self.rawCapture = PiRGBArray(self.camera, size = IM_SIZE)
        time.sleep(0.5)
        
        self.joy = xbox.Joystick()
        

    # Loop over camera frames
    def videoLoop(self):
        start = time.time()
        i = 0
        for frame in self.camera.capture_continuous(self.rawCapture, format="rgb", use_video_port=True):
            # convert img as Array
            image = frame.array
            # take a pic
            if self.label[0] != -1 and self.snap == True:
                if time.time() - start > self.delay:
                    im = Image.fromarray(image, 'RGB')
                    t_stamp = time.time()
                    picname = "/home/pi/Documents/patate_v2/get_datas/datas" + str(self.label[0]) + "_" + str(self.label[1]) + "_" + str(t_stamp) + ".jpg"
                    im.save(picname)
                    print(str(i) + " - snap : " + picname)
                    i += 1
                    start = time.time()
            # Clean image before the next comes
            self.rawCapture.truncate(0)
            
            if self.joy.A():                   #Test state of the A button (1=pressed, 0=not pressed)
                self.pwm.set_pwm(0, 0, 0)
                self.pwm.set_pwm(1, 0, 0)
                print("Stop")
                return
            self.controls()

    def controls(self):
        self.speed  = self.joy.rightTrigger() # Get floats values between 0 to 1.0
        self.speed = 322 - (self.speed * 17)
        self.speed = round(self.speed, 0)
        self.speed = int(self.speed)

        self.direction = self.joy.leftX() # Get floats values between -1.0 to 1.0
        self.direction = (self.direction * 80) + 320
        self.direction = round(self.direction, 0)
        self.direction = int(self.direction)

        self.label[0] = self.speed
        self.label[1] = self.direction
        self.pwm.set_pwm(0, 0, self.direction)
        self.pwm.set_pwm(1, 0, self.speed)

if __name__ == "__main__":
    print("Press Ctrl+C to start/stop...")
    try:
      while True:
        pass
    except KeyboardInterrupt:
      pass
    controler = Controler()
    controler.videoLoop()
    controler.joy.close()
    controler.camera.close()
    
    



