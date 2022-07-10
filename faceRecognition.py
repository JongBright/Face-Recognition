import cv2
import face_recognition




image = face_recognition.load_image_file('Test files/Jong Bright.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
testImage = face_recognition.load_image_file('Test files/Jong Bright (3).jpeg')
testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(image)[0]
encodedImage = face_recognition.face_encodings(image)[0]
cv2.imwrite('test.jpg', encodedImage)
cv2.rectangle(image, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

testfaceLoc = face_recognition.face_locations(testImage)[0]
encodedtestImage = face_recognition.face_encodings(testImage)[0]
cv2.rectangle(testImage, (testfaceLoc[3], testfaceLoc[0]), (testfaceLoc[1], testfaceLoc[2]), (0, 255, 0), 2)


result = face_recognition.compare_faces([encodedImage], encodedtestImage)
faceDistance = face_recognition.face_distance([encodedImage], encodedtestImage)
print(faceDistance)
print(result)
cv2.putText(testImage, f'{result}: {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)



cv2.imshow('Image', image)
cv2.imshow('testImage', testImage)
cv2.waitKey(0)
