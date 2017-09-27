import numpy as np
import cv2

frames = input('Input Frame Numbers: ')

data = np.load('10.npy')

for i in range(int(frames)):

    cv2.imshow('window',data[i][0])
    cv2.waitKey(0)
    if i == int(frames)-1:
        cv2.destroyAllWindows()
