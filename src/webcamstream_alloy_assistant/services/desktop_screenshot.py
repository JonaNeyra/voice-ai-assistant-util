import base64
import time

import numpy
from threading import Lock, Thread
from cv2 import imencode, cvtColor, COLOR_RGB2BGR
from PIL import ImageGrab

class DesktopScreenshot:
    def __init__(self):
        self.screenshot = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            screenshot = ImageGrab.grab()
            screenshot = cvtColor(numpy.array(screenshot), COLOR_RGB2BGR)

            self.lock.acquire()
            self.screenshot = screenshot
            self.lock.release()

            time.sleep(0.1)

    def read(self, encode=False):
        self.lock.acquire()
        screenshot = self.screenshot.copy() if self.screenshot is not None else None
        self.lock.release()

        if encode and screenshot is not None:
            _, buffer = imencode(".jpeg", screenshot)
            return base64.b64encode(buffer)

        return screenshot

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
