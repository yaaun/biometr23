import argparse
from pathlib import Path

import PIL.Image

import biometr23.minutiae as minutiae
import biometr23.threskel as threskel
from biometr23.color_minmap import minutiae_map_to_RGB_Image


def main():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("image_file", type=str)

    args = parser.parse_args()

    in_path = Path(args.image_file)
    img = PIL.Image.open(in_path)

    if img.mode not in {"1", "L"}:
        img = img.convert("L")

    img_skel = threskel.process(img)
    minutiae_map = minutiae.minutiae_map_filtered(img_skel)

    out_name = Path(in_path.stem + "_minutiae").with_suffix(".png")
    out_img = minutiae_map_to_RGB_Image(minutiae_map)

    out_img.save(out_name)


if __name__ == "__main__":
    main()
