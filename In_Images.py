import cv2
from random import randrange
img_file = "C:\\Users\\adama\\PycharmProjects\\133971.png"
classifier_file = "car_detector.xml"

img = cv2.imread(img_file)
car_tracker = cv2.CascadeClassifier(classifier_file)
grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
car_coordinates = car_tracker.detectMultiScale(grayscale)
for (x,y,w,h) in car_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),2)
cv2.imshow("Detecting Cars...",img)
cv2.waitKey()


print("Done")

#"C:\Users\adama\PycharmProjects\stock-photo-city-people-street-busy-london-crowded-streetlight-double-decker-street-photography-495a4378-aad9-4e6a-bb56-317ebfdc55d8.jpg"