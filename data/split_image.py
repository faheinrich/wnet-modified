import numpy as np


def split_image(img, square_region_size):
    subregions = []

    rows = int(np.ceil(img.shape[0] / square_region_size))
    cols = int(np.ceil(img.shape[1] / square_region_size))
    for i in range(0, img.shape[0], square_region_size):
        for j in range(0, img.shape[1], square_region_size):
            sub = img[i:i + square_region_size, j:j + square_region_size]
            if sub.shape[0] != square_region_size or sub.shape[1] != square_region_size:
                if len(img.shape) == 3:
                    square_sub = np.zeros((square_region_size, square_region_size, 3), dtype=img.dtype)
                elif len(img.shape) == 2:
                    square_sub = np.zeros((square_region_size, square_region_size), dtype=img.dtype)
                square_sub[:sub.shape[0], :sub.shape[1]] = sub
                subregions.append(square_sub)
            else:
                subregions.append(sub)

    return subregions, rows, cols


def stitch_image(subregions: list, rows, cols, imgHeight, imgWidth):
    size = subregions[0].shape[0]
    if len(subregions[0].shape) == 3:
        img = np.zeros((rows*size, cols*size, 3), dtype=subregions[0].dtype)
    elif len(subregions[0].shape) == 2:
        img = np.zeros((rows * size, cols * size), dtype=subregions[0].dtype)
    else:
        assert False, "img wrong"

    for i in range(0, rows):
        for j in range(0, cols):
            sub = subregions[(i*cols)+j]
            img[i*size:(i*size)+size, j*size:(j*size)+size] = sub
    return img[:imgHeight, :imgWidth]