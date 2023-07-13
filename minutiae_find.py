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
    overlay, overlay3, overlay_grey_bool_RGBA, overlay_grey_RGBA
from biometr23.util import path_list_from_argslist


def main():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("image_file_or_dir", type=str, nargs="+", help="The path to the image file to find minutiae in. "
        "Posix-style globs will be resolved.")
    parser.add_argument("-f", "--outputForm", choices=["overlay3", "pixels"], default="overlay3",
                        help="Form of output. 'overlay3' blends the input image, its skeleton and the minutiae pixels. "
                        "'pixels' outputs only minutiae locations as colored pixels (red = termination, green = bifurcation), "
                             "with all other pixels black.")
    parser.add_argument("-o", "--outputDir", default=None,
                        help="Directory where to place output files. If not set, uses the same directory as the input file.")
    parser.add_argument("--debugDir", "--dd", default=".",
                        help="Debug file directory, used only when verbose mode is on. Defaults to current working directory")
    parser.add_argument("--dryRun", "--dry", action="store_true", default=False,
                        help="Run program, but do not read nor write any files.")
    parser.add_argument("-T", "--threshold", type=int, dest="thresholdHint", default=None)
    parser.add_argument("-S", "--shift", type=int, dest="thresholdShift", default=None)
    parser.add_argument("--skeletonAlpha", type=int, default=127,
                        help="Set the transparency value (from 0 to 255) for the fingerprint skeleton in 'overlay3' output mode.")
    parser.add_argument("-v", action="count", default=0,
                        help="Activate verbose debugging mode. Can repeat 'v' up to 2 times to increase verbosity level.")

    args = parser.parse_args()

    dbgPath = Path(args.debugDir)
    dbgLv = args.v

    img_files = path_list_from_argslist(args.image_file_or_dir, ".")

    if dbgLv >= 2:
        print("img_files = " + str(img_files))


    for i, in_fname in enumerate(img_files):
        if args.dryRun:
            break

        in_path = Path(in_fname)
        img = PIL.Image.open(in_path)

        if img.mode not in {"1", "L"}:
            img = img.convert("L")

        try:
            # note that this returns an ndarray, not PIL.Image
            img_bin = threskel.autothreshold(img, thresh_hint=args.thresholdHint, thresh_shift=args.thresholdShift)
        except RuntimeError as e:
            warnings.warn("autothreshold failed, skipping '" + str(in_fname) + "'; error message: " + str(e))
            continue

        if dbgLv > 0:
            skimage.io.imsave(dbgPath / f"Seq{i:03d}_DbgImg010_autothreshold.png", skimage.util.img_as_ubyte(img_bin))

        img_skel = threskel.skeletonize_clean(img_bin)
        if dbgLv > 0:
            skimage.io.imsave(dbgPath / f"Seq{i:03d}_DbgImg020_skeletonize_clean.png", skimage.util.img_as_ubyte(img_skel))

        minutiae_map = minutiae.minutiae_map_filtered(img_skel)

        if args.outputForm == "overlay3":
            out_name = Path(in_path.stem + "_overlay3").with_suffix(".png")
            over_img = minutiae_map_to_RGBA_Image(minutiae_map)
            out_img = overlay_grey_bool_RGBA(img, img_skel, over_img, args.skeletonAlpha)
            #out_img = overlay3(img.convert("RGBA"), PIL.Image.fromarray(img_skel * 255).convert("RGBA"), over_img)
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
