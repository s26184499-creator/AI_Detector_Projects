import cv2
from random import randrange

#load some pretrained data on face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

#choose an image to detect face
# img=cv2.imread('ammy.jpg')
img=cv2.imread('datasets\images\people02.jpg')

#convert image to greyscaled image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

cv2.rectangle(grayscaled_img, (182,  64), (182+195, 64+195), (0,255,0), 2)

# (x,y,w,h) = face_coordinates[0]
# cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

print(face_coordinates)

#display image
cv2.imshow('clever program face detector', img)

#wait to close
cv2.waitKey()
