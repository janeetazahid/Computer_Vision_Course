#import all libraries 
import cv2
import matplotlib.pyplot as plt
import numpy as np
#import image we're using
car_plates=cv2.imread('car_plate.jpg')

#function to display images bigger and change color mapping
def display(img):
    fig=plt.figure(figsize=(10,8)) #set figure size
    ax=fig.add_subplot(111)
    new_image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change color mapping
    ax.imshow(new_image)
#display original image 
display(car_plates)

#function to blur lincense, based on detection of license plate
def detect_and_blur_plate(img):
    plate_blur=car_plates.copy()
    roi=car_plates.copy()
    cas=cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    plate_rect=cas.detectMultiScale(plate_blur,scaleFactor=1.2,minNeighbors=2)
    for (x,y,w,h) in plate_rect:
        roi=roi[y:y+h,x:x+w]         #find the proper region of interest
        blurred_roi=cv2.medianBlur(roi,7)
        plate_blur[y:y+h,x:x+w]=blurred_roi
    return plate_blur

#store the result as a new image 
result = detect_and_blur_plate(car_plates)

#display result
display(result)
