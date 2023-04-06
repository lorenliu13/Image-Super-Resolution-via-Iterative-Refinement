import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
NPY_EXTENSIONS = ['.npy']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    """
    Get path from images
    """
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def is_numpy_file(filename):
    """
    04/05/2023
    Return whether the file extension is .npy
    """
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)


def get_paths_from_numpys(path):
    """
    04/05/2023
    Get path from npy files
    """
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_numpy_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor() # converts a PIL image or numpy.ndarray to a Pytorch tensor
# convert the img to tensor data type
# rescale the pixel values from original value (0-255) to floating point between 0 and 1, basically, divide each pixel value by 255
# if there is 3 channel image, change the order from height, width, channel to channel, height, width
hflip = torchvision.transforms.RandomHorizontalFlip() # horizontally flips the given image randomly with default prob 0.5
def transform_augment(img_list, split='val', min_max=(0, 1)):
    """

    img_list: a list of image to apply augmentation
    split: whether the data is for training or validation
    min_max: specifying the minimum and maximum values to normalize the data
    """
    imgs = [totensor(img) for img in img_list] # conver the input images to tensors
    if split == 'train':
        # if phrase is train, apply the horizontal flip augmentation
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs] # normalize the image tensors by scaling them with min_max
    return ret_img
