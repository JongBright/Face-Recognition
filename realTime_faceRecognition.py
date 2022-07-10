import cv2
import face_recognition
import numpy as np
import os
import datetime
import random



faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

path = 'Known_Faces'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cls in myList:
    currImage = cv2.imread(f'{path}/{cls}')
    images.append(currImage)
    classNames.append(os.path.splitext(cls)[0])

def findEncodings(imageList):
    encodedList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (len(face_recognition.face_encodings(img)))!=0:
            encodedImg = face_recognition.face_encodings(img)[0]
            encodedList.append(encodedImg)

    return encodedList


encodedKnownFace = findEncodings(images)
print('****************** Encoding completed ********************')

check = []
check2 = []
check3 = []
identicalFaces = []
namesIdenticalFaces = []
possibleFacesIndexes = []
mainFaceHolder = set()
mainFaceIndexHolder = set()
names = []



cap = cv2.VideoCapture(0)

while True:

    frame, imgS = cap.read()

    img = cv2.resize(imgS, (0,0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurrFrame = face_recognition.face_locations(img)
    encodedCurrFrame = face_recognition.face_encodings(img, facesCurrFrame)

    for encodedFace, faceLoc in zip(encodedCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodedKnownFace, encodedFace)
        faceDistance = face_recognition.face_distance(encodedKnownFace, encodedFace)

        possibleFaces = []
        for i in faceDistance:
            if (round(i, 2)<=0.38):
                possibleFaces.append(i)


        #face has been recognized
        if len(possibleFaces) != 0:

            matchIndex = np.argmin(possibleFaces)
            if matches[matchIndex]:



                if len(possibleFaces) == 1:
                    faceProb = possibleFaces[0]
                    faceIndex = np.argmin(faceDistance)
                    name = classNames[faceIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    if name=='BOSS':
                        cv2.rectangle(imgS, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(imgS, (x1, y2-35), (x2, y2), (255, 0, 0), cv2.FILLED)
                    else:
                        cv2.rectangle(imgS, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(imgS, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)

                    cv2.putText(imgS, name, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                    print(f'Face recognized| name: {name} accuracy: {round(100-(faceProb*100), 2)}%')





                if len(possibleFaces) > 1:
                    for faceProbability in possibleFaces:
                        faceIndex = list(faceDistance).index(faceProbability)
                        faceImage = images[faceIndex]
                        possibleFaceIndex = possibleFaces.index(faceProbability)
                        faceName = classNames[faceIndex]
                        if faceImage in np.array(identicalFaces):
                           print('Already exist')
                        else:
                           if len(check2) < 2:
                               identicalFaces.append(faceImage)
                               namesIdenticalFaces.append(faceName)
                               possibleFacesIndexes.append(possibleFaceIndex)
                               check2.append(1)


                    if len(identicalFaces)>0:
                        if len(namesIdenticalFaces)>0:
                            for face in namesIdenticalFaces:
                                if '-' not in face:
                                    mainFace = face
                                    mainFaceHolder.add(mainFace)
                                    print(f'MainFace: {mainFace}')

                        if len(mainFaceHolder)>0:
                            if len(possibleFacesIndexes) > 0:
                                print(possibleFacesIndexes)
                                print(possibleFaces)
                                print(namesIdenticalFaces)
                                mainFaceIndex = possibleFacesIndexes[namesIdenticalFaces.index(list(mainFaceHolder)[0])]
                                mainFaceIndexHolder.add(mainFaceIndex)

                        if len(check3) == 0:
                            x = random.choice(range(len(identicalFaces)))
                            check3.append(x)
                        else:
                            testFaceIndex = check3[0]
                            #cv2.imshow('mainFace', identicalFaces[mainFaceIndex])
                            #cv2.imshow('test2', identicalFaces[testFaceIndex])
                            if len(mainFaceIndexHolder)>0:
                                mainFaceEncode = face_recognition.face_encodings(identicalFaces[list(mainFaceIndexHolder)[0]])[0]
                                testFaceEncode = face_recognition.face_encodings(identicalFaces[testFaceIndex])[0]

                                if list(mainFaceIndexHolder)[0] != testFaceIndex:
                                    result = face_recognition.compare_faces([mainFaceEncode], testFaceEncode)

                                    if result[0] == True:
                                        print('The detected faces are identical')

                                        faceProbb = possibleFaces[list(mainFaceIndexHolder)[0]]
                                        possibleName = list(mainFaceHolder)[0].upper()
                                        y1, x2, y2, x1 = faceLoc
                                        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                                        if possibleName=='BOSS':
                                            cv2.rectangle(imgS, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                            cv2.rectangle(imgS, (x1, y2-35), (x2, y2), (255, 0, 0), cv2.FILLED)
                                        else:
                                            cv2.rectangle(imgS, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.rectangle(imgS, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)

                                        cv2.putText(imgS, possibleName, (x1+6, y2-6), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                                        print(f'Face recognized| name: {possibleName} accuracy: {round(100-(faceProbb*100), 2)}%')


                                else:
                                    check3.clear()



        #face has not been recognized
        if len(possibleFaces) == 0:
            print('*************** Face not recognized ******************')
           



    cv2.imshow('Image', imgS)
    key = cv2.waitKey(1)
    if key==27:
        break




cap.release()
cv2.destroyAllWindows()
