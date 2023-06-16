import numpy as np

# Taken and adapted to NumPy from Przemysław Pastuszka crossing_number.py (https://github.com/przemekpastuszka/biometrics)
def minutiae_at(pixels, y0, x0):
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    pxvals = np.array([pixels[y0 + dy, x0 + dx] for dy, dx in neighbours], dtype=np.int8)
    crossings2 = np.abs(np.diff(pxvals)).sum()

    if pixels[y0, x0]:
        if crossings2 == 2:
            return 1
        if crossings2 == 6:
            return 3
    return 0

# Taken from Przemysław Pastuszka crossing_number.py (https://github.com/przemekpastuszka/biometrics)
# Adapted to numpy array image representation
def minutiae_map(img):
    (max_y, max_x) = img.shape
    minutiae_locs = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(1, max_y - 1):
        for j in range(1, max_x - 1):
            minutiae = minutiae_at(img, i, j)
            if minutiae is not None:
                minutiae_locs[i, j] = minutiae

    return minutiae_locs

