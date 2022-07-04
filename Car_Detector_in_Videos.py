import cv2
from random import randrange
vid_file = "C:\\Users\\adama\\Downloads\\videoplayback (1).mp4"
classifier_file = "car_detector.xml"
classifier_file2 = "haarcascade_fullbody.xml"


vid = cv2.VideoCapture(vid_file)
car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(classifier_file2)
while True:
    successful_frame_read, frame = vid.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_coordinates = car_tracker.detectMultiScale(grayscale)
    pedestrian_coordinates =pedestrian_tracker.detectMultiScale(grayscale)
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    for (x,y,w,h) in pedestrian_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Detecting Cars and Pedestrians...",frame)
    key=cv2.waitKey(1)
    if key== 81 or key == 113:
        break

vid.release()
print("Done")