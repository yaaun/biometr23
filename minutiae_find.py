import argparse
from pathlib import Path

import PIL.Image
import skimage.io
import skimage.util

import biometr23.minutiae as minutiae
import biometr23.threskel as threskel
from biometr23.output_minmap import minutiae_map_to_RGB_Image, overlay, overlay3


def main():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("image_file", type=str, nargs="+", help="The path to the image file to find minutiae in.")
    parser.add_argument("--outputForm", choices=["overlay", "overlay3", "pixels"], default="overlay")
    parser.add_argument("--debugPath", default=".")
    parser.add_argument("-v", action="count")

    args = parser.parse_args()

    dbgPath = Path(args.debugPath)
    dbgLv = args.v

    for i, in_fname in enumerate(args.image_file):
        in_path = Path(in_fname)
        img = PIL.Image.open(in_path)

        if img.mode not in {"1", "L"}:
            img = img.convert("L")

        img_bin = threskel.autothreshold(img) # note that this returns an ndarray, not PIL.Image
        if dbgLv > 0:
            skimage.io.imsave(dbgPath / f"Seq{i:03d}_DbgImg010_autothreshold.png", skimage.util.img_as_ubyte(img_bin))

        img_skel = threskel.skeletonize_clean(img_bin)
        if dbgLv > 0:
            skimage.io.imsave(dbgPath / f"Seq{i:03d}_DbgImg020_skeletonize_clean.png", skimage.util.img_as_ubyte(img_skel))

        minutiae_map = minutiae.minutiae_map_filtered(img_skel)

        if args.outputForm == "overlay3":
            out_name = Path(in_path.stem + "_overlay3").with_suffix(".png")
            over_img = minutiae_map_to_RGB_Image(minutiae_map)
            out_img = overlay3(img.convert("RGB"), PIL.Image.fromarray(img_skel * 255).convert("RGB"), over_img)
        elif args.outputForm == "overlay":
            out_name = Path(in_path.stem + "_overlay").with_suffix(".png")
            over_img = minutiae_map_to_RGB_Image(minutiae_map)
            out_img = overlay(img.convert("RGB"), over_img)
        elif args.outputForm == "pixels":
            out_img = minutiae_map_to_RGB_Image(minutiae_map)

        out_img.save(out_name)


if __name__ == "__main__":
    main()
