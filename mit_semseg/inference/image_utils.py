from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

"""
image utils
"""
def imshow(im):
    plt.imshow(im)
    plt.show()


def round2nearest_multiple(x, p):
    # Round x to the nearest multiple of p and x' >= x
    return ((x - 1) // p + 1) * p

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


def img_transform(img):
    img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = img_normalize(torch.from_numpy(img.copy()))
    return img

def visualize_pred_mask(image, mask_dict, threshold):
    h, w = image.shape[:2]

    cls2colors = {
        0: (255,0,0), #eye
        1: (0,255,0),  #eyebrow
        2: (0,0,255)
    }
    mask_rgb = np.ones(shape=(h,w,3), dtype=np.uint8) * 255
    for cls_id, mask_cls in mask_dict.items():
        color = cls2colors[cls_id]
        mask_rgb[mask_cls > threshold] = color

    visualize_image = 0.5 * image.astype(np.float) + 0.5 * mask_rgb.astype(np.float)
    visualize_image = np.clip(visualize_image.astype(np.uint8), 0, 255)

    return visualize_image