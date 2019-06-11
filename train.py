import cv2
import threading
import os
import time
import pickle
import argparse
import imutils
from imutils import paths
from imutils.video import VideoStream
import face_recognition

flag = True
flag2 = False
i_global = 0
k = 0
xml_ready=False

def graying():
    while flag:
        for k in range(6):
            print("thread"+str(k))
            if os.path.isfile("./unknown/" + str(k) + ".jpg"):
                gray = cv2.imread("./unknown/"+str(k)+".jpg")
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("./unknown/"+str(k)+"cinza.jpg", gray)
        time.sleep(1)


def encode_faces():
    while flag:
        if(i_global==5):
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
                print("[INFO] processing image {}/{}".format(j + 1,
                                                         len(imagePaths)))
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
            xml_ready=True


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle",
	help="path to serialized db of facial encodings")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

while True:
    print("[INFO] loading encodings...")
    data = pickle.loads(open(args["encodings"], "rb").read())
    writer = None
    time.sleep(2.0)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    t = threading.Thread(target=encode_faces)
    t.start()
    while not xml_ready:
        if not os.path.exists("./dataset/unknown"):
            os.makedirs("./dataset/unknown")
        ret, frame = cap.read()
        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=900)
        r = frame.shape[1] / float(rgb.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
                                                model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
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
            else:
                cv2.imwrite("./dataset/unknown/"+str(int(i_global))+".jpg", frame)
                print("nova foto salva de unknown")
                i_global += 1

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        # check to see if we are supposed to display the output frame to
        # the screen
        if args["display"] > 0:
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    flag=False
    cap.release()
    cv2.destroyAllWindows()
    xml_ready=False

