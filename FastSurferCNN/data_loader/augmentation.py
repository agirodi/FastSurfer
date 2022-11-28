# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
from numbers import Number

import numpy as np
import torch


##
# Transformations for evaluation
##
class ToTensorTest(object):
    """
    Convert np.ndarrays in sample to Tensors

    Functions:
        __call__: converts image
    """

    def __call__(self, img):
        """
        Convertes the image to float within range [0, 1] and make it torch compatible

        Args:
            img (ndarray): image to be conformed

        Returns:
            ndarray: conformed image
        """

        img = img.astype(np.float32)

        # Normalize and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return img


class ZeroPad2DTest(object):
    """
    Pad the input with zeros to get output size

    Functions:
        __init__: constructor
        pad: pads zeroes of image
        call: calls _pad()
    """

    def __init__(self, output_size, pos='top_left'):
        """
        Constructor

        Args:
            output_size (float, Tuple[float, float]): size of the output image either as float or tuple of two floats
            pos (str): position
        """
        if isinstance(output_size, Number):
            output_size = (int(output_size), ) * 2
        self.output_size = output_size
        self.pos = pos

    def _pad(self, image):
        """
        pads zeros of the input image [help]

        Args:
            image (ndarray): The image to pad

        Returns:
            ndarray: original image with padded zeros
        """

        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros(self.output_size, dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros(self.output_size + (c,), dtype=image.dtype)

        if self.pos == 'top_left':
            padded_img[0: h, 0: w] = image

        return padded_img

    def __call__(self, img):
        """
        calls the _pad() function

        Args:
            img (ndarray):the image to pad

        Returns:
            ndarray: original image with padded zeros
        """

        img = self._pad(img)

        return img


##
# Transformations for training
##
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.

    Functions:
        __call__: converts image
    """

    def __call__(self, sample):
        """[help]
        Convertes the image to float within range [0, 1] and make it torch compatible

        Args:
            sample (): sample object

        Returns:
            Dict[str, Tensor]: [help]
        """

        img, label, weight, sf = sample['img'], sample['label'], sample['weight'], sample['scale_factor']

        img = img.astype(np.float32)

        # Normalize image and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return {'img': torch.from_numpy(img),
                'label': torch.from_numpy(label),
                'weight': torch.from_numpy(weight),
                'scale_factor': torch.from_numpy(sf)}


class ZeroPad2D(object):
    """
    pads zeroes

    Functions:
        __init__: constructor
        _pad: pads zeroes of image
        __call__: cals _pad for sample
    """

    def __init__(self, output_size, pos='top_left'):
        """
        Initializes position and output_size (as Tuple[float])

        Args:
            output_size (float, Tuple[float]):
            pos (str): position to put the input
        """
        if isinstance(output_size, Number):
            output_size = (int(output_size), ) * 2
        self.output_size = output_size
        self.pos = pos

    def _pad(self, image):
        """
        pads zeros of the input image [help]

        Args:
            image (ndarray): the image to pad

        Returns:
            ndarray: original image with padded zeros
        """

        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros(self.output_size, dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros(self.output_size + (c,), dtype=image.dtype)

        if self.pos == 'top_left':
            padded_img[0: h, 0: w] = image

        return padded_img

    def __call__(self, sample):
        img, label, weight, sf = sample['img'], sample['label'], sample['weight'], sample['scale_factor']

        img = self._pad(img)
        label = self._pad(label)
        weight = self._pad(weight)

        return {'img': img, 'label': label, 'weight': weight, 'scale_factor': sf}


class AddGaussianNoise(object):
    """
    add noise to sample

    Functions:
        __call__: adds noise to scale factor
    """

    def __init__(self, mean=0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        """
        adds gaussian noise to scalefactor
        :param: {img, label, weight, scale_factor}: sample: sample to add noise
        :return: {img, label, weight, scale_factor}: sample with noise
         """

        img, label, weight, sf = sample['img'], sample['label'], sample['weight'], sample['scale_factor']
        # change 1 to sf.size() for isotropic scale factors (now same noise change added to both dims)
        sf = sf + torch.randn(1) * self.std + self.mean
        return {'img': img, 'label': label, 'weight': weight, 'scale_factor': sf}


class AugmentationPadImage(object):
    """
    Pad Image with either zero padding or reflection padding of img, label and weight
    :function: __call: pads zeroes
    """

    def __init__(self, pad_size=((16, 16), (16, 16)), pad_type="edge"):

        assert isinstance(pad_size, (int, tuple))

        if isinstance(pad_size, int):

            # Do not pad along the channel dimension
            self.pad_size_image = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            self.pad_size_mask = ((pad_size, pad_size), (pad_size, pad_size))

        else:
            self.pad_size = pad_size

        self.pad_type = pad_type

    def __call__(self, sample):
        """
        pads zeroes of sample image, label and weight
        :param: AugementationPadImage: self: the object
        :param: {img, label, weight, scale_factor}: sample: sample image and data
        """

        img, label, weight, sf = sample['img'], sample['label'], sample['weight'], sample['scale_factor']

        img = np.pad(img, self.pad_size_image, self.pad_type)
        label = np.pad(label, self.pad_size_mask, self.pad_type)
        weight = np.pad(weight, self.pad_size_mask, self.pad_type)

        return {'img': img, 'label': label, 'weight': weight, 'scale_factor': sf}


class AugmentationRandomCrop(object):
    """
    Randomly Crop Image to given size
    :function: __init__: constructor
    :function: __call__: crops the Augmentation
    """

    def __init__(self, output_size, crop_type='Random'):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            self.output_size = output_size

        self.crop_type = crop_type

    def __call__(self, sample):
        """[help]
        crops the augmentation randomly
        :param: AugmentationRandomCrop: self: the object
        :param: {img, label, weight, scale_factor}: sample: sample image with data
        """
        img, label, weight, sf = sample['img'], sample['label'], sample['weight'], sample['scale_factor']

        h, w, _ = img.shape

        if self.crop_type == 'Center':
            top = (h - self.output_size[0]) // 2
            left = (w - self.output_size[1]) // 2

        else:
            top = np.random.randint(0, h - self.output_size[0])
            left = np.random.randint(0, w - self.output_size[1])

        bottom = top + self.output_size[0]
        right = left + self.output_size[1]

        # print(img.shape)
        img = img[top:bottom, left:right, :]
        label = label[top:bottom, left:right]
        weight = weight[top:bottom, left:right]

        return {'img': img, 'label': label, 'weight': weight, 'scale_factor': sf}
