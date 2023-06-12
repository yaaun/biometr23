import matplotlib
matplotlib.use("TkAgg")

import skimage.io
import skimage.morphology

def main():
    inimg = skimage.io.imread("BW_swirl.png")
    skelImg = skimage.morphology.skeletonize(inimg)
    skimage.io.imshow(skelImg)
    skimage.io.show()

if __name__ == "__main__":
    main()
