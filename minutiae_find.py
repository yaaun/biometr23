import argparse
from pathlib import Path
import warnings

import PIL.Image
import numpy as np
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
    parser.add_argument("-f", "--outputForm", choices=["overlay3", "pixels", "polar"], default="overlay3",
                        help="Form of output. 'overlay3' (default) blends the input image, its skeleton and the minutiae pixels. "
                        "'pixels' outputs only minutiae locations as colored pixels (red = termination, green = bifurcation), "
                             "with all other pixels black. 'polar' generates a tab-separated table in a text file"
                             "containing polar coordinates of minutiae in the first two columns, relative to the binarized"
                             "image's center of mass, and a third column with '1' indicating a termination and '3' a bifurcation.")
    parser.add_argument("-o", "--outputDir", default=None,
                        help="Directory where to place output files. If not set, uses the same directory as the input file.")
    parser.add_argument("--debugDir", "--dd", default=".",
                        help="Debug file directory, used only when verbose mode is on. Defaults to current working directory")
    parser.add_argument("--dryRun", "--dry", action="store_true", default=False,
                        help="Run program, but do not read nor write any files.")
    parser.add_argument("-T", "--threshint", type=int, dest="thresholdHint", default=None,
                        help="The threshold hint, or the initial mean value for fitting the gaussian that will "
                             "determine the binarization threshold")
    parser.add_argument("-S", "--shift", type=int, dest="thresholdShift", default=None,
                        help="The threshold shift, relative to the fitted gaussian mean")
    parser.add_argument("--skeletonAlpha", type=int, default=127,
                        help="Set the transparency value (from 0 to 255) for the fingerprint skeleton in 'overlay3' output mode.")
    parser.add_argument("-v", action="count", default=0,
                        help="Activate verbose debugging mode. Can repeat 'v' up to 2 times to increase verbosity level.")

    args = parser.parse_args()

    dbgPath = Path(args.debugDir)
    dbgLv = args.v

    img_files = path_list_from_argslist(args.image_file_or_dir, ".")

    if dbgLv >= 1:
        print("img_files = " + str([*map(str, img_files)]))


    for i, in_fname in enumerate(img_files):
        if args.dryRun:
            break

        in_path = Path(in_fname)
        dbgPrefix = in_path.stem
        img = PIL.Image.open(in_path)

        if img.mode not in {"1", "L"}:
            img = img.convert("L")

        try:
            # note that this returns an ndarray, not PIL.Image
            img_bin = threskel.autothreshold(img, thresh_hint=args.thresholdHint, thresh_shift=args.thresholdShift)
        except RuntimeError as e:
            warnings.warn("autothreshold failed, skipping '" + str(in_fname) + "'; error message: " + str(e))
            continue

        if dbgLv >= 2:
            skimage.io.imsave(dbgPath / f"{dbgPrefix}_DbgImg010_autothreshold.png", skimage.util.img_as_ubyte(img_bin))

        img_bin_clean = threskel.clean_binary(img_bin)

        if dbgLv >= 2:
            skimage.io.imsave(dbgPath / f"{dbgPrefix}_DbgImg015_clean_binary.png", skimage.util.img_as_ubyte(img_bin_clean))

        img_skel = threskel.skeletonize(img_bin_clean)
        if dbgLv >= 2:
            skimage.io.imsave(dbgPath / f"{dbgPrefix}_DbgImg020_skeletonize.png", skimage.util.img_as_ubyte(img_skel))

        img_skel_clean = threskel.clean_skeleton(img_skel)
        if dbgLv >= 2:
            skimage.io.imsave(dbgPath / f"{dbgPrefix}_DbgImg025_clean_skeleton.png",
                              skimage.util.img_as_ubyte(img_skel_clean))
        #
        # if dbgLv >= 2:
        #     minutiae_map_unclean = minutiae.minutiae_map(img_skel_clean)
        #     skimage.io.imsave(dbgPath / f"Seq{i:03d}_DbgImg100_.png",
        #                      skimage.util.img_as_ubyte(img_skel_clean))

        minutiae_map = minutiae.minutiae_map_filtered(img_skel_clean)

        out_img = None
        if args.outputForm == "overlay3":
            out_name = Path(in_path.stem + "_overlay3").with_suffix(".png")
            over_img = minutiae_map_to_RGBA_Image(minutiae_map)
            out_img = overlay_grey_bool_RGBA(img, img_skel, over_img, args.skeletonAlpha)
            #out_img = overlay3(img.convert("RGBA"), PIL.Image.fromarray(img_skel * 255).convert("RGBA"), over_img)
        elif args.outputForm == "pixels":
            out_name = Path(in_path.stem + "_pixels").with_suffix(".png")
            out_img = minutiae_map_to_RGB_Image(minutiae_map)
        elif args.outputForm == "polar":
            centerOfMass = minutiae.calc_CoM(img_bin_clean)
            tableTermins = minutiae.bin_matrix_to_polar_coords(minutiae_map == 1, *centerOfMass)
            tableBifurcs = minutiae.bin_matrix_to_polar_coords(minutiae_map == 3, *centerOfMass)

            tableTermins = np.concatenate((tableTermins, np.broadcast_to(1, (1, tableTermins.shape[1]))), axis=0)
            tableBifurcs = np.concatenate((tableBifurcs, np.broadcast_to(3, (1, tableBifurcs.shape[1]))), axis=0)

            out_name = Path(in_path.stem + "_polar").with_suffix(".txt")


        if args.outputDir is None:
            outputPath = in_path.parent / out_name
        else:
            outputPath = Path(args.outputDir)
            if not outputPath.exists():
                outputPath.mkdir()
            outputPath = outputPath / out_name


        if out_img is not None:
            out_img.save(outputPath)
        else:
            with open(outputPath, "wt", encoding="utf-8") as fout:
                fmt = ["%.2f", "%.2f", "%d"]
                np.savetxt(fout, tableTermins.T, fmt, delimiter="\t")
                np.savetxt(fout, tableBifurcs.T, fmt, delimiter="\t")

if __name__ == "__main__":
    main()
