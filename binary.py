'''
Code from last year's version of jet_detect(). Uses thresholding to create binary
image, on which cv2.HoughLines() is run. This code takes two images: a
calibration image and a second image, and attempts jet detection on both of them.
Only the strongest line found by the Hough transform is displayed.
'''

import numpy as np
import cv2
import math
# from matplotlib import pyplot as plt


def image_stats(img):
    return img.mean(), img.std()

# from cam_utils
def jet_detect(img, calibratemean, calibratestd, x):
    mean, std = image_stats(img)
    
    # compare mean & calibratemean
    if (mean < calibratemean * 0.8) or (mean > calibratemean * 1.2):
        print('no jet')

    for c in range(x):
        try:
            binary = (img / (mean + 2 * std * 0.90 ** c)).astype(np.uint8)
            # binary = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 50, 2)
            # binary = cv2.bitwise_not(imagem)
            lines = cv2.HoughLines(binary, 1, np.radians(0.25), 30)
            rho, theta = lines[0][0]
            if (theta > math.radians(45)) and (theta < math.radians(135)):
                print('invalid jet')
            if (get_jet_width(img, rho, theta) * pxsize > 0.1):
                print('invalid jet')
            # for rho, theta in lines[0]:
                # jetValid = true
                # if (theta > math.radians(70)):
                #     jetValid = false
		# width = get_jet_width(binary, rho, theta)
                # if (width > [x]):
                #     jetValid = false
                # if (jetValid == false):
                # reject jet
        except Exception:
            print(c)
            continue
        else:
            # show binary image
            # cv2.imshow('binary', binary)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return rho, theta
    raise ValueError('unable to detect jet')

# get calibration image
# img = cv2.imread('lines.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('white.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('black.jpg', cv2.IMREAD_GRAYSCALE)
filename = input("calibration image: ")
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# show original image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# choose how many times to attempt jet detection
x = int(input("c: "))

calimean, calistd = image_stats(img)
print(calimean)
print(calistd)
rho, theta = jet_detect(img, calimean, calistd, x)

print(rho)
print(theta)

# draw line on original image
a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
x1 = int(x0 + 1000 * (-b))
y1 = int(y0 + 1000 * (a))
x2 = int(x0 - 1000 * (-b))
y2 = int(y0 - 1000 * (a))
lined = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.line(lined, (x1, y1), (x2, y2), (0, 0, 255), 2)

# show binary image
# cv2.imshow('binary', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save binary image
# cv2.imwrite('images/binary.jpeg', binary)

# show lined image
cv2.imshow('lined', lined)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(lined)
# plt.show()

# save lined image
# cv2.imwrite('images/lined.jpeg', lined)

# get second image
filename2 = input("second image: ")
img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)

# show second image
cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# choose how many times to attempt jet detection
x = int(input("c: "))

rho2, theta2 = jet_detect(img2, calimean, calistd, x)

print(rho2)
print(theta2)

# draw line on second  image
a2 = np.cos(theta2)
b2 = np.sin(theta2)
x0b = a2 * rho2
y0b = b2 * rho2
x1b = int(x0b + 1000 * (-b2))
y1b = int(y0b + 1000 * (a2))
x2b = int(x0b - 1000 * (-b2))
y2b = int(y0b - 1000 * (a2))
lined2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
cv2.line(lined2, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)

# show lined second image
cv2.imshow('lined', lined2)
cv2.waitKey(0)
cv2.destroyAllWindows()

