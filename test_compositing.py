import PIL.Image
import numpy as np


def main():
    eye5 = np.eye(5, dtype=bool)
    eye5f = np.fliplr(eye5)

    im1 = PIL.Image.fromarray(np.uint8(255) * eye5, "L")
    im2 = PIL.Image.fromarray(np.uint8(255) * eye5f, "L")

    im1.save("im1.png")
    im2.save("im2.png")

    mask = PIL.Image.new("L", im1.size, 0)
    mask.putpixel((2, 2),  255)
    mask.putpixel((1, 1), 127)

    im12 = PIL.Image.composite(im1, im2, mask)
    im12.save("im12.png")


if __name__ == "__main__":
    main()