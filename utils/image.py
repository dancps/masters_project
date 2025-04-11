import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from masters.utils.logger import Loggir
from datetime import datetime as dt
try:
    from termcolor import colored
except ImportError:
    def colored(inp,*s):
        return inp
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(inp,*s):
        return inp

selflogger = Loggir()

def path_checker(path):
    """
    Checks the given path(s) for image files and analyzes their dimensions.
    This function processes directories or individual image files. For directories, 
    it recursively traverses all subdirectories to find image files with extensions 
    `.png`, `.jpg`, or `.jpeg`. It calculates and displays the count, minimum, 
    maximum, and median dimensions of 2D and 3D images found in the directory. 
    For individual image files, it displays their dimensions.
    Args:
        path (str or list of str): A single path or a list of paths to check. 
                                   Paths can be directories or image files.
    Behavior:
        - If the path is a directory, it processes all image files within the directory 
          and its subdirectories.
        - If the path is an image file, it displays the dimensions of the image.
        - If the path is neither a valid directory nor an image file, it prints an 
          error message.
    Prints:
        - For directories:
            - The count of 2D and 3D images.
            - The minimum, maximum, and median dimensions for each type of image.
        - For individual image files:
            - The dimensions of the image.
        - Error messages for invalid paths.
    Note:
        - The function uses `os` for file and directory operations.
        - The function uses `matplotlib.pyplot` to read image files.
        - The function assumes that valid image files have extensions `.png`, `.jpg`, or `.jpeg`.
    """
    
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                print(colored(f"Processing directory: {root}", 'cyan'))
                image_dimensions = []
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        image = plt.imread(file_path)
                        # print(f"Image: {file_path}, Dimensions: {image.shape}")
                        image_dimensions.append(image.shape)
                if image_dimensions:
                    for dim_size in [2,3]:
                        dim_filter = [i for i in image_dimensions if len(i) == dim_size]
                        if len(dim_filter) == 0:
                            print(f"{dim_size}D - Count: 0")
                        else:
                            print(f"{dim_size}D - Count: {len(dim_filter)}")
                            nparr = np.array(dim_filter)
                            min_dim = nparr.min(axis=0)
                            max_dim = nparr.max(axis=0)
                            median_dim = np.median(nparr, axis=0)
                            print(f"  - Min Dimensions   : {min_dim}\n  - Max Dimensions   : {max_dim}\n  - Median Dimensions: {median_dim}")
        elif os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = plt.imread(path)
            print(f"Image: {path}, Dimensions: {image.shape}")
        else:
            print(f"Path {path} is not a valid image or directory containing images.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs = "+", type=str, help='input')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    # parser.add_argument('--multichoice', choices=['a', 'b', 'c'], nargs='+', type=str, help='multiple types of arguments. May be called all at the same time.')
    args = parser.parse_args()

    paths = args.input


if(__name__=='__main__'):
    init=dt.now()
    main()
    end=dt.now()
    print('Elapsed time: {}'.format(end-init))