import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import PIL.Image

import biometr23.threskel
from biometr23.util import path_list_from_argslist


def outputPlot(outPath, histCounts, gaussParams=None):
    xs = np.arange(0, 256)

    if gaussParams is not None:
        ys_gauss = biometr23.threskel.gauss(xs, *gaussParams)

    with mpl.rc_context():
        fig, ax = plt.subplots()

        ax.bar(xs, histCounts, 1, color="0.5")

        if gaussParams is not None:
            ax.plot(xs, ys_gauss)

        fig.savefig(outPath)




def main():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument("image_file_or_dir", type=str, nargs="+",
                        help="The path to the image file to find minutiae in.")

    parser.add_argument("-f", "--outputForm", choices=["plot", "table"], default="plot")
    parser.add_argument("-o", "--outputDir", default=None)
    parser.add_argument("--debugDir", "--dd", default=".")
    parser.add_argument("--dryRun", "--dry", action="store_true", default=False)
    parser.add_argument("-v", action="count", default=0)

    args = parser.parse_args()

    img_paths = path_list_from_argslist(args.image_file_or_dir)

    for impath in img_paths:
        im = PIL.Image.open(impath).convert("L")

        histList = im.histogram()

        if args.outputDir is None:
            outDirPath = impath.parent
        else:
            outDirPath = args.outputDir

        if args.outputForm == "plot":
            outName = impath.stem + "_HistPlot"
            gaussParams = biometr23.threskel.fit_gauss_hist256(histList)
            outputPlot(outDirPath / outName, histList, gaussParams)




if __name__ == "__main__":
    main()
