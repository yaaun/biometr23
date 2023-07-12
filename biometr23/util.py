from pathlib import Path


def path_list_from_argslist(argslist, baseDir="."):
    paths = []
    for i, img_pathstr_file_or_dir in enumerate(argslist):
        globResult = Path(baseDir).glob(img_pathstr_file_or_dir)

        paths.extend(globResult)

    return paths
