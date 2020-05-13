import cv2 as cv
import skimage.feature

img = cv.imread('../data/exemplePozitive/caltech_web_crop_00013.jpg')
hog = skimage.feature.hog(img, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L1',
                          feature_vector=False)
# cv.imshow('img', hist)
# cv.waitKey(0)
# cv.destroyAllWindows()
print(hog)
