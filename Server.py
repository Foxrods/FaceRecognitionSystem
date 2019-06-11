from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import imutils
import cv2

imageHub = imagezmq.ImageHub()
frameDict = {}
while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    cv2.putText(frame, rpiName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frameDict[rpiName] = frame
    montages = build_montages(frameDict.values(), (w, h), (2, 1))
    for (i, montage) in enumerate(montages):
        cv2.imshow("Camera ({})".format(i), montage)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
