import cv2
import threading
import os
import time
import pickle
import argparse
import imutils
import imagezmq
import datetime
import pytz
import requests
from imutils import paths
from imutils import build_montages
import face_recognition

#Flags globais de controle de thread
flag_treinar=False
terminate_threads=False
fotos_tiradas=0
f = open("n_pessoas.txt", "r")
n_pessoas = int(f.read())
f.close()
frameDict = {}
imageHub = imagezmq.ImageHub()
API_link = "link"
avistamentos = {}
qtd_rostos_detec = {}
data = 0
def cadastrar():
    global flag_treinar
    global n_pessoas
    nome = input("Nome do individuo")
    print("cadastro")
    os.rename("./dataset/individuo"+n_pessoas+"/","./dataset/"+nome)
    flag_treinar = True

#Função de codificação dos rosto, gera o arquivo xml encodings.pickle
def face_encode():
    global flag_treinar
    global fotos_tiradas
    global n_pessoas
    global data
    while not terminate_threads:
        if flag_treinar:
            # construct the argument parser and parse the arguments
            ap = argparse.ArgumentParser()
            ap.add_argument("-i", "--dataset", default="dataset",
                            help="path to input directory of faces + images")
            ap.add_argument("-e", "--encodings", default="encodings.pickle",
                            help="path to serialized db of facial encodings")
            ap.add_argument("-d", "--detection-method", type=str, default="hog",
                            help="face detection model to use: either `hog` or `cnn`")
            args = vars(ap.parse_args())

            # grab the paths to the input images in our dataset
            print("[INFO] quantifying faces...")
            imagePaths = list(paths.list_images(args["dataset"]))

            # initialize the list of known encodings and known names
            knownEncodings = []
            knownNames = []

            # loop over the image paths
            for (j, imagePath) in enumerate(imagePaths):
                # extract the person name from the image path
                print("[INFO] processing image {}/{}".format(j + 1, len(imagePaths)))
                name = imagePath.split(os.path.sep)[-2]

                # load the input image and convert it from RGB (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)

                # loop over the encodings
                for encoding in encodings:
                    # add each encoding + name to our set of known names and
                    # encodings
                    knownEncodings.append(encoding)
                    knownNames.append(name)

            # dump the facial encodings + names to disk
            print("[INFO] serializing encodings...")
            data = {"encodings": knownEncodings, "names": knownNames}
            f = open(args["encodings"], "wb")
            f.write(pickle.dumps(data))
            f.close()
            flag_treinar=False
            fotos_tiradas=0
            n_pessoas+=1
            f = open("n_pessoas.txt","w")
            f.write(str(n_pessoas))
            f.close()
            data = pickle.loads(open(args["encodings"], "rb").read())

def face_recog(args, frame):
    global fotos_tiradas
    global flag_treinar
    global n_pessoas
    global rpiName
    global rpiName_antes
    global lista_pessoas_antes
    global lista_pessoas_antes2
    global avistamentos
    global qtd_rostos_detec
    global data

    writer = None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgb = imutils.resize(frame, width=500)
    r = frame.shape[1] / float(rgb.shape[1])
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    qtd_rostos_detec.update({rpiName:len(boxes)})
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    loop_counter = 0
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
            if not(name in avistamentos):
                avistamentos[name] = rpiName
                data_hora = datetime.datetime.now(pytz.timezone('America/Sao_Paulo'))
                dados = {'nome': name,
                         'camera': rpiName,
                         'data': str(data_hora.day) + "/" + str(data_hora.month) + "/" + str(data_hora.year),
                         'hora': str(data_hora.hour) + ":" + str(data_hora.minute) + ":" + str(data_hora.second)}
                #r=requests.post(API_link, data = dados)
                print(dados)
            elif(avistamentos[name]!=rpiName):
                avistamentos[name] = rpiName
                data_hora = datetime.datetime.now(pytz.timezone('America/Sao_Paulo'))
                dados = {'nome': name,
                         'camera': rpiName,
                         'data': str(data_hora.day) + "/" + str(data_hora.month) + "/" + str(data_hora.year),
                         'hora': str(data_hora.hour) + ":" + str(data_hora.minute) + ":" + str(data_hora.second)}
                # r=requests.post(API_link, data = dados)
                print(dados)

        else:
            if not (name in names):
                loop_counter +=1
                name = "Unknown" + str(n_pessoas)
                n_pessoas += 1
                if not os.path.exists('./dataset/individuo'+name.replace("Unknown",'')):
                    os.makedirs('./dataset/individuo'+name.replace("Unknown",''))
                for(top, right, bottom, left) in boxes:
                    face_frame=frame[top:bottom, left:right]
                    cv2.imwrite("./dataset/individuo"+name.replace("Unknown",'')+"/" + str(fotos_tiradas+1) + ".jpg", face_frame)
                print("nova foto salva de um desconhecido: "+name)
                list = os.listdir("./dataset/individuo"+name.replace("Unknown",''))  #conta quantos fotos tem na pasta
                fotos_tiradas = len(list)
                print(fotos_tiradas)
                if fotos_tiradas == 40:
                    flag_treinar = True
        # update the list of names
        names.append(name)

    n_pessoas-=loop_counter

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
       top = int(top * r)
       right = int(right * r)
       bottom = int(bottom * r)
       left = int(left * r)
        # draw the predicted face name on the image
       cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
       y = top - 15 if top - 15 > 15 else top + 15
       cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return frame


def main():
    global n_pessoas
    global frameDict
    global imageHub
    global rpiName
    global data

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--encodings", default="encodings.pickle",
                    help="path to serialized db of facial encodings")
    ap.add_argument("-y", "--display", type=int, default=1,
                    help="whether or not to display output frame to screen")
    ap.add_argument("-d", "--detection-method", type=str, default="hog",
                    help="face detection model to use: either `hog` or `cnn`")
    args = vars(ap.parse_args())
    data = pickle.loads(open(args["encodings"], "rb").read())
    print("[INFO] loading encodings...")
    t = threading.Thread(target=face_encode)
    t.start()
    while True:
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')
        frame = imutils.resize(frame, width=500)
        frame = face_recog(args, frame)
        (h, w) = frame.shape[:2]
        cv2.putText(frame, rpiName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frameDict[rpiName] = frame

        montages = build_montages(frameDict.values(), (w, h), (2, 1))
        for montage in montages:
            cv2.imshow("Montage", montage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('n'):
            cadastrar()

    cv2.destroyAllWindows()
    global terminate_threads
    terminate_threads = True

if __name__ == '__main__':
    main()