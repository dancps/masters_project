import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from decimal import localcontext, Decimal
from termcolor import colored

def generate_dataset_metadata(folder_path):
    """This method generates a metadata df for a given folder

    Args:
        folder_path (_type_): _description_
    """

    if any(os.path.isdir(os.path.join(folder_path, entry)) for entry in os.listdir(folder_path)):
        raise ValueError("Expected to be a single folder. Found subdirectories inside the folder.")
    
    files = [f for f in sorted(os.listdir(folder_path))]
    print(f"{colored(folder_path, attrs= ['bold'])}: {len(files)} files")

    # Remove non image files
    ALLOWED_FORMATS = ['png', 'jpg']
    image_files = [f for f in files if os.path.splitext(f)[-1].replace(".", "") in ALLOWED_FORMATS]
    print(f"  Removing non-images: {len(image_files)} files left ({len(image_files)-len(files)})")

    # Get size of images
    files_metadata = []
    for f in image_files:
        full_path = os.path.join(folder_path, f)
        img = plt.imread(full_path)
        _, ext = os.path.splitext(f)
        if len(img.shape)==2:
            w = img.shape[0]
            h = img.shape[1]
            c = None
            img_type = '2ch'
            with localcontext(prec=5) as ctx:
                ratio = Decimal(w)/Decimal(h)
        elif len(img.shape)==3:
            w = img.shape[0]
            h = img.shape[1]
            c = img.shape[2]
            img_type = '3ch'
            with localcontext(prec=5) as ctx:
                ratio = Decimal(w)/Decimal(h)
        else:
            raise ValueError(f"Image {full_path} with not a valid shape: {img.shape}")
        metadata = {
            'folder': folder_path,
            'image_name': f,
            'full_path': full_path,
            'w': w,
            'h': h,
            'c': c,
            'shape': img.shape,
            'image_ratio': ratio,
            'format': ext.replace(".", ""),
            'type': img_type
        }
        files_metadata.append(metadata)

    df = pd.DataFrame(files_metadata)
    # print(df)

    print(df.agg({
        'w': ['min', 'max', 'median'],
        'h': ['min', 'max', 'median'],
        'c': ['min', 'max', 'median'],
        'image_ratio': ['min', 'max', 'median'],
    }))
    print()
    return df

def get_metadata_of_classes_folder(dataset_folder):
    classes = [item for item in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, item))] 

    df = pd.DataFrame()
    for img_class in classes:
        folder_path = os.path.join(dataset_folder, img_class)
        tempdf = generate_dataset_metadata(folder_path)
        tempdf["class"] = img_class

        df = pd.concat([df, tempdf])
    
    print(df)
    print(df.agg({
        'w': ['min', 'max', 'median'],
        'h': ['min', 'max', 'median'],
        'c': ['min', 'max', 'median'],
        'image_ratio': ['min', 'max', 'median'],
    }))

    return df




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input')
    parser.add_argument('-o', '--output', default='.', type=str, help='output')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: The input path '{args.input}' is not a valid directory.")
        exit(1)

    folder = args.input

    get_metadata_of_classes_folder(folder)