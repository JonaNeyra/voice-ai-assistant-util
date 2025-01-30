import base64
from threading import Lock, Thread
from cv2 import VideoCapture, imencode


class WebcamStream:
    def __init__(self):
        self.thread = None
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
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
            _, frame = self.stream.read()
            if self.lock.acquire(blocking=False):  # Intenta bloquear sin esperar
                self.frame = frame
                self.lock.release()

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.release()
