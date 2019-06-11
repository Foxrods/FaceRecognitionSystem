from imutils.video import VideoStream
import imutils
import cv2
import imagezmq
import socket
import time

server = "192.168.1.111"
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(server))
rpiName = socket.gethostname()
webcam = VideoStream(src=0).start()
webcam1 = VideoStream(src=1).start()
time.sleep(2.0)

while True:
    frame = webcam.read()
    frame1 = webcam1.read()
    sender.send_image(rpiName+" cam0", frame)
    sender.send_image(rpiName + " cam0", frame1)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
webcam.stop()
webcam1.stop()