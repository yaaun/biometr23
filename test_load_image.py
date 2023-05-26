import numpy as np
import matplotlib
import matplotlib.pyplot
matplotlib.use("TkAgg")
import skimage.io

# Taken from Przemysław Pastuszka crossing_number.py (https://github.com/przemekpastuszka/biometrics)
def minutiae_at(pixels, i, j):
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    values = [pixels[i + k][j + l] for k, l in neighbours]

    crossings = 0
    for k in range(0, 8):
        crossings += abs(values[k] - values[k + 1])
    crossings /= 2

    if pixels[i][j] == 1:
        if crossings == 1:
            return "ending"
        if crossings == 3:
            return "bifurcation"
    return "none"

# Taken from Przemysław Pastuszka crossing_number.py (https://github.com/przemekpastuszka/biometrics)
# Adapted to numpy array image representation
def calculate_minutiaes(img):
    (max_y, max_x) = img.shape
    minutiae_locs = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

    #colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}
    #ellipse_size = 2

    for i in range(1, max_y - 1):
        for j in range(1, max_x - 1):
            minutiae = minutiae_at(img, i, j)
            if minutiae != "none":
                minutiae_locs[i, j] = [255, 0, 0, 255]

    return minutiae_locs

def run():
    img = skimage.io.imread("skel_1_8.png", as_gray=True)
    assert img.ndim == 2 # powinien to być obraz czarno-biały
    print(f"{img.ndim=}")
    #skimage.io.imshow(img, plugin="matplotlib")
    # Convert down to 0, 1 range
    #img.flat[img.flat]
    matplotlib.pyplot.imshow(img//255, cmap="binary", interpolation="nearest")

    minutiae_locs = calculate_minutiaes(img)

    #matplotlib.pyplot.imshow(minutiae_locs, cmap="cool", alpha=0.5)
    matplotlib.pyplot.imshow(minutiae_locs)
    matplotlib.pyplot.show(block=True)

if __name__ == "__main__":
    run()