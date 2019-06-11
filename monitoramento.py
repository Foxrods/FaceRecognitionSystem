import cv2
import os
import pickle
import imutils
import imagezmq
import datetime
import pytz
import requests
from imutils import paths
from imutils import build_montages
import face_recognition

fotos_tiradas = 0
frameDict = {}
avistamentos = {}
data = 0
f = open("n_pessoas.txt", "r")
n_pessoas = int(f.read())
f.close()


def face_recog(frame):
    global data
    global n_pessoas
    global frameDict
    global avistamentos
    global rpiName
    f = open("n_pessoas.txt", "r")
    n_pessoas = int(f.read())
    f.close()
    names = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = frame.shape[1] / float(rgb.shape[1])
    boxes = face_recognition.face_locations(rgb, model="hog")
    if len(boxes)>=1:
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

                if (not (name in avistamentos)) or (avistamentos[name] != rpiName):
                    avistamentos[name] = rpiName
                    data_hora = datetime.datetime.now(pytz.timezone('America/Sao_Paulo'))
                    dados = {'nome': name,
                         'camera': rpiName,
                         'data': str(data_hora.day) + "/" + str(data_hora.month) + "/" + str(data_hora.year),
                         'hora': str(data_hora.hour) + ":" + str(data_hora.minute) + ":" + str(data_hora.second)}
                    # r=requests.post(API_link, data = dados)
                    print(dados)

            if name == "Unknown":
                name = "Unknown" + str(n_pessoas)
            names.append(name)
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 128, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 128, 0), 2)

    return frame


def main():
    global data
    global frameDict
    global rpiName
    imageHub = imagezmq.ImageHub(open_port='tcp://192.168.1.111:5555')
    #imageHub2 = imagezmq.ImageHub(open_port='tcp://192.168.1.111:5557')
    data = pickle.loads(open("encodings.pickle", "rb").read())
    while True:
        (rpiName, frame) = imageHub.recv_image()
        #(rpiName2, frame2) = imageHub2.recv_image()
        imageHub.send_reply(b'OK')
        #imageHub2.send_reply(b'OK')
        frame = imutils.resize(frame, width=500)
        #frame2 = imutils.resize(frame2, width=500)
        frame = face_recog(frame)
        #frame2 = face_recog(frame2)
        (h, w) = frame.shape[:2]
        cv2.putText(frame, rpiName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(frame2, rpiName2, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frameDict[rpiName] = frame
        #frameDict[rpiName2] = frame2
        montages = build_montages(frameDict.values(), (w, h), (2, 1))
        for montage in montages:
            cv2.imshow("Cãmeras", montage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()