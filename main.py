import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file("elon.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("test.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 0), 2)

result = face_recognition.compare_faces([encodeElon], encodeTest)
print(result)

cv2.imshow("Elon", imgElon)
cv2.imshow("Test", imgTest)
cv2.waitKey(0)

# imgElonSmall = cv2.resize(imgElon, (0, 0), fx=0.25, fy=0.25)

