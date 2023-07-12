import argparse
from pathlib import Path
import warnings

import PIL.Image
import skimage.io
import skimage.util

import biometr23.minutiae as minutiae
import biometr23.threskel as threskel
from biometr23.output_minmap import \
    minutiae_map_to_RGB_Image, minutiae_map_to_RGBA_Image,\
    overlay, overlay3, overlay_grey_bool_RGB


def main():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("image_file_or_dir", type=str, nargs="+", help="The path to the image file to find minutiae in.")
    parser.add_argument("--outputForm", choices=["overlay", "overlay3", "pixels"], default="overlay")
    parser.add_argument("--outputDir", default=None)
    parser.add_argument("--debugDir", default=".", dest="debugPath")
    parser.add_argument("--dryRun", action="store_true", default=False)
    parser.add_argument("-v", action="count", default=0)

    args = parser.parse_args()

    dbgPath = Path(args.debugPath)
    dbgLv = args.v

    img_files = []
    for i, img_pathstr_file_or_dir in enumerate(args.image_file_or_dir):
        p = Path(img_pathstr_file_or_dir)
        globResult = Path(".").glob(img_pathstr_file_or_dir)

        img_files.extend(globResult)

    if dbgLv >= 2:
        print("img_files = " + repr(img_files))


    for i, in_fname in enumerate(img_files):
        if args.dryRun:
            break

        in_path = Path(in_fname)
        img = PIL.Image.open(in_path)

        if img.mode not in {"1", "L"}:
            img = img.convert("L")

        try:
            img_bin = threskel.autothreshold(img) # note that this returns an ndarray, not PIL.Image
        except RuntimeError as e:
            warnings.warn("autothreshold failed, skipping; error message: " + str(e))
            continue

        if dbgLv > 0:
            skimage.io.imsave(dbgPath / f"Seq{i:03d}_DbgImg010_autothreshold.png", skimage.util.img_as_ubyte(img_bin))

        img_skel = threskel.skeletonize_clean(img_bin)
        if dbgLv > 0:
            skimage.io.imsave(dbgPath / f"Seq{i:03d}_DbgImg020_skeletonize_clean.png", skimage.util.img_as_ubyte(img_skel))

        minutiae_map = minutiae.minutiae_map_filtered(img_skel)

        if args.outputForm == "overlay3":
            out_name = Path(in_path.stem + "_overlay3").with_suffix(".png")
            #over_img = minutiae_map_to_RGB_Image(minutiae_map)
            #out_img = overlay3(img.convert("RGB"), PIL.Image.fromarray(img_skel * 255).convert("RGB"), over_img)
            over_img = minutiae_map_to_RGBA_Image(minutiae_map)
            #out_img = overlay_grey_bool_RGB(img, img_skel, over_img)
            out_img = overlay3(img.convert("RGBA"), PIL.Image.fromarray(img_skel * 255).convert("RGBA"), over_img)
        elif args.outputForm == "overlay":
            out_name = Path(in_path.stem + "_overlay").with_suffix(".png")
            over_img = minutiae_map_to_RGB_Image(minutiae_map)
            out_img = overlay(img.convert("RGB"), over_img)
        elif args.outputForm == "pixels":
            out_img = minutiae_map_to_RGB_Image(minutiae_map)

        if args.outputDir is None:
            out_img.save(in_path.parent / out_name)
        else:
            outputPath = Path(args.outputDir)
            if not outputPath.exists():
                outputPath.mkdir()
            out_img.save(outputPath / out_name)


if __name__ == "__main__":
    main()
