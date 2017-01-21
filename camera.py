import numpy as np
import cv2
from matplotlib import pyplot as plt

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

while(True):
    # Capture frame-by-frame
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()

    # Our operations on the frame come here
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray1,gray2)

    # Display the resulting frame
    cv2.imshow('frame1',gray1)
    cv2.imshow('frame2',gray2)
    #plt.imshow(disparity, 'gray')
    #plt.imsave(disparity, 'yolo.png')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap1.release()
cap2.release()
cv2.destroyAllWindows()