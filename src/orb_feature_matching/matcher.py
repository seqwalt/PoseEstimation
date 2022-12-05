import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time

print("reading images")
img1 = cv.imread('cubeThing.png')          # queryImage
img2 = cv.imread('cubeThing2.png') # trainImage
print("done reading images")
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
start = time.time()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print("here")
# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
print("here1")
matches = bf.match(des1,des2)
print("here2")
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
end = time.time()
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print(end - start)


plt.imshow(img3),plt.show()
