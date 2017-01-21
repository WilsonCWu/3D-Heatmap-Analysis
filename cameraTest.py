import numpy as np
import cv2
from matplotlib import pyplot as plt

cap1 = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 28.0, (1280,720))

while(True):
    # Capture frame-by-frame
    ret, frame1 = cap1.read()

    # Our operations on the frame come here
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame1',gray1)
    out.write(frame1)
    #cplt.imshow(disparity, 'gray')
    #plt.imsave(disparity, 'yolo.png')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap1.release()
out.release()
cv2.destroyAllWindows()