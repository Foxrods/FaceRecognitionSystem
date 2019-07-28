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
import shutil

fotos_tiradas = 0
frameDict = {}
avistamentos = {}
data = 0
n_pessoas = len(os.listdir("./dataset/"))+1
Dict_of_Unknowns = {}
list_values = []


def face_recog(frame):
    global data
    global n_pessoas
    global frameDict
    global avistamentos
    global rpiName
    global Dict_of_Unknowns
    global list_values
    global fotos_tiradas
    names = []
    n_pessoas = len(os.listdir("./dataset/"))+1
    loop_counter = 0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = frame.shape[1] / float(rgb.shape[1])
    boxes = face_recognition.face_locations(rgb, model="hog")

    if len(boxes)>=1:
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.52)
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
                name = "Unknown"+str(n_pessoas)
                n_pessoas+=1
                loop_counter += 1 #conta quantos Unknows há detectado nas cameras

            names.append(name)
        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)
            if "Unknown" in name:
                face_frame = frame[top:bottom, left:right]
                rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                box = face_recognition.face_locations(rgb_face_frame, model="hog")
                minor_distance = 1
                k=0
                k_menor=0
                if len(box) != 0:
                    #saving the first photo on the new folder
                    if not os.path.exists('./desconhecidos/individuo'+name.replace("Unknown",'')):
                        os.makedirs('./desconhecidos/individuo'+name.replace("Unknown",''))
                        cv2.imwrite("./desconhecidos/individuo" + name.replace("Unknown", '') + "/" + str(fotos_tiradas + 1) + ".jpg",face_frame)
                        encoding_of_unknown = face_recognition.face_encodings(rgb_face_frame,box)[0]
                        Dict_of_Unknowns[name] = encoding_of_unknown
                    elif os.path.exists('./desconhecidos/individuo'+name.replace("Unknown",'')) and not(name in Dict_of_Unknowns.keys()):
                        imageloaded = face_recognition.load_image_file('./desconhecidos/individuo'+name.replace("Unknown",'')+"/1.jpg")
                        encoding_of_unknown = face_recognition.face_encodings(imageloaded)[0]
                        #print(encoding_of_unknown)
                        Dict_of_Unknowns[name] = encoding_of_unknown
                        #print(Dict_of_Unknowns[name].tolist())
                    else: #checking if the new unknown x corresponds to the previous one
                        new_encoding_of_unknown = face_recognition.face_encodings(rgb_face_frame, box)[0]
                        for key,encode in Dict_of_Unknowns.items():
                            face_distances = face_recognition.face_distance([encode],new_encoding_of_unknown)
                            print(face_distances)
                            if face_distances<minor_distance:
                                minor_distance=face_distances
                                k_menor = key
                        #index = int(name.replace("Unknown",''))+k_menor
                        name = k_menor
                        print(name)
                        if minor_distance>[0.60]:
                            k_pessoas = len(os.listdir("./dataset/")) + len(os.listdir("./desconhecidos/"))+1
                            name = "Unknown" + str(k_pessoas)
                            if not os.path.exists('./desconhecidos/individuo' + name.replace("Unknown", '')):
                                os.makedirs('./desconhecidos/individuo' + name.replace("Unknown", ''))
                                cv2.imwrite("./desconhecidos/individuo" + name.replace("Unknown", '') + "/" + str(
                                    fotos_tiradas + 1) + ".jpg", face_frame)
                                encoding_of_unknown = face_recognition.face_encodings(rgb_face_frame, box)[0]
                                Dict_of_Unknowns[name] = encoding_of_unknown
                        lista = os.listdir("./desconhecidos/individuo" + name.replace("Unknown",''))
                        fotos_tiradas = len(lista)
                        print(fotos_tiradas)
                        if fotos_tiradas < 4 and minor_distance < 0.55:
                            #k_pessoas = len(os.listdir("./dataset/"))+1
                            if not(os.path.exists("./desconhecidos/individuo" + name.replace("Unknown", '') + "/" + str(fotos_tiradas + 1) + ".jpg")):
                                cv2.imwrite("./desconhecidos/individuo" + name.replace("Unknown", '') + "/" + str(fotos_tiradas + 1) + ".jpg", face_frame)
                            print("nova foto salva de um desconhecido: " + name)
                            print(fotos_tiradas)
                        elif fotos_tiradas >= 4:
                            k_pessoas = len(os.listdir("./dataset/")) + 1
                            print("Individuo" + str(k_pessoas) + " pronto para ser cadastrado")
                            source = "./desconhecidos/individuo" + str(k_pessoas)+"/"
                            os.makedirs('./dataset/individuo' + str(k_pessoas))
                            dest1 = "./dataset/individuo" + str(k_pessoas)
                            files = os.listdir(source)
                            for f in files:
                                shutil.move(source + f, dest1)
                            os.rmdir(source) #delete empty folder in desconhecidos
                            del Dict_of_Unknowns[name]
                            face_encode(dest1)
            cv2.rectangle(frame, (left, top), (right, bottom), (64, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (64, 255, 0), 2)

    n_pessoas -= loop_counter
    return frame


def face_encode(path):
    global n_pessoas
    global fotos_tiradas
    global data
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(path))
    knownEncodings = data["encodings"]
    knownNames = data["names"]
    for (j, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(j + 1, len(imagePaths)))
        name = path
        name = name.replace("./dataset/",'')
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    print("[INFO] serializing encodings...")
    #data = {"encodings": knownEncodings, "names": knownNames}
    #print(data)
    data["encodings"]=knownEncodings
    data["names"]=knownNames
    print(data)
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    fotos_tiradas = 0
    data = pickle.loads(open("encodings.pickle","rb").read())
    print("Cadastramento concluído")


def main():
    global data
    global frameDict
    global rpiName
    #imageHub = imagezmq.ImageHub(open_port='tcp://192.168.1.111:5555')
    #imageHub2 = imagezmq.ImageHub(open_port='tcp://192.168.1.111:5557')
    data = pickle.loads(open("encodings.pickle", "rb").read())
    capture = cv2.VideoCapture(0)
    #capture2 = cv2.VideoCapture(1) inicia a captura de video de uma segunda webcam
    while True:
        #(rpiName, frame) = imageHub.recv_image()
        #(rpiName2, frame2) = imageHub2.recv_image()
        #imageHub.send_reply(b'OK')
        #imageHub2.send_reply(b'OK')
        ret, frame = capture.read()
        #ret2, frame2 = capture2.read() faz a captura dos frames na outra webcam
        frame = imutils.resize(frame, width=500)
        #frame2 = imutils.resize(frame2, width=500) reescalona
        frame = face_recog(frame)
        #frame2 = face_recog(frame2) reconhece os rostos na segunda webcam
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
