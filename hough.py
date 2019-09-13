'''
Takes an image and performs canny(), hough_line(), and hough_line_peaks() on it.
Displays the Hough space and every line found in the Hough transform.
'''

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

# use canny and hough_lines from skimage rather than cv2

# get calibration image
#img = cv2.imread('lines.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('white.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('black.jpg', cv2.IMREAD_GRAYSCALE)
filename = input("image: ")
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# show original image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# use canny
binary = canny(img, sigma=2, use_quantiles=True, low_threshold=0.9, high_threshold=0.99)

# show canny image
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3), sharex=True, sharey=True)
ax1.imshow(img, cmap=plt.cm.gray)
ax2.imshow(binary, cmap=plt.cm.gray)
plt.show()

# use hough_line
h, angles, d = hough_line(binary)
#print(h)
#print(angles)
#print(d)

#show hough_line
fix, axes = plt.subplots(1, 2, figsize=(7, 4))
axes[0].imshow(img, cmap=plt.cm.gray)
#axes[1].imshow(h, cmap=plt.cm.bone, extent=(np.rad2deg(angles[-1]),
#                np.rad2deg(angles[0]), d[-1], d[0]))
axes[1].imshow(np.log(1 + h), 
               extent=[np.rad2deg(angles[-1]), 
               np.rad2deg(angles[0]), 
               d[-1], d[0]], cmap=plt.cm.gray, aspect=1/1.5)
plt.show()

# use hough_line_peaks
results = hough_line_peaks(h, angles, d)
print(results)

# show lined image
fix, axes = plt.subplots(1, 2, figsize=(7, 4))
axes[0].imshow(img, cmap=plt.cm.gray)
axes[1].imshow(img, cmap=plt.cm.gray)
for _, angle, dist in zip(*results):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
    axes[1].plot((0, img.shape[1]), (y0, y1), '-r'))
plt.show()

# show lined image
#cv2.imshow('lined', lined)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# plt.imshow(lined)
# plt.show()

