from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_bool, morphology
import cv2


def main():
    with Image.open("3_2.tif") as img:
        img.load()
        im_array = np.asarray(img)

    thresh = cv2.adaptiveThreshold(im_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 10)
    cv2.imshow("Adaptive thresholding", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    thresh_img = Image.fromarray(thresh)
    edge_enhance = thresh_img.filter(ImageFilter.EDGE_ENHANCE)
    no_holes = morphology.remove_small_holes(img_as_bool(edge_enhance), 15)
    no_small_obj = morphology.remove_small_objects(no_holes, 5, connectivity=2)
    plt.imshow(no_small_obj, cmap='gray', interpolation='nearest')
    plt.show()
    skeleton = morphology.medial_axis(no_small_obj)
    plt.imshow(skeleton, cmap='gray', interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()