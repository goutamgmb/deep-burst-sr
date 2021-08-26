import random
import numpy as np
import math
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvisf


class Transform:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, *transforms):
        if len(transforms) == 1 and isinstance(transforms[0], (list, tuple)):
            transforms = transforms[0]
        self.transforms = transforms
        self._valid_inputs = ['image']
        self._valid_args = ['joint', 'new_roll']
        self._valid_all = self._valid_inputs + self._valid_args

    def __call__(self, **inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        for v in inputs.keys():
            if v not in self._valid_all:
                raise ValueError('Incorrect input \"{}\" to transform. Only supports inputs {} and arguments {}.'.format(v, self._valid_inputs, self._valid_args))

        joint_mode = inputs.get('joint', True)
        new_roll = inputs.get('new_roll', True)

        if not joint_mode:
            out = zip(*[self(**inp) for inp in self._split_inputs(inputs)])
            return tuple(list(o) for o in out)

        out = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        # out = inputs

        for t in self.transforms:
            out = t(**out, joint=joint_mode, new_roll=new_roll)
        if len(var_names) == 1:
            return out[var_names[0]]
        # Make sure order is correct
        return tuple(out[v] for v in var_names)

    def _split_inputs(self, inputs):
        var_names = [k for k in inputs.keys() if k in self._valid_inputs]
        split_inputs = [{k: v for k, v in zip(var_names, vals)} for vals in zip(*[inputs[vn] for vn in var_names])]
        for arg_name, arg_val in filter(lambda it: it[0]!='joint' and it[0] in self._valid_args, inputs.items()):
            if isinstance(arg_val, list):
                for inp, av in zip(split_inputs, arg_val):
                    inp[arg_name] = av
            else:
                for inp in split_inputs:
                    inp[arg_name] = arg_val
        return split_inputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TransformBase:
    """ Class for applying various image transformations."""
    def __init__(self):
        self._valid_inputs = ['image']
        self._valid_args = ['new_roll']
        self._valid_all = self._valid_inputs + self._valid_args
        self._rand_params = None

    def __call__(self, **inputs):
        # Split input
        input_vars = {k: v for k, v in inputs.items() if k in self._valid_inputs}
        input_args = {k: v for k, v in inputs.items() if k in self._valid_args}

        # Roll random parameters for the transform
        if input_args.get('new_roll', True):
            rand_params = self.roll()
            if rand_params is None:
                rand_params = ()
            elif not isinstance(rand_params, tuple):
                rand_params = (rand_params,)
            self._rand_params = rand_params

        outputs = dict()
        for var_name, var in input_vars.items():
            if var is not None:
                params = self._rand_params
                if isinstance(var, (list, tuple)):
                    outputs[var_name] = [getattr(self, 'transform_' + var_name)(x, *params) for x in var]
                else:
                    outputs[var_name] = getattr(self, 'transform_' + var_name)(var, *params)
        return outputs

    def _get_image_size(self, inputs):
        im = None
        for var_name in ['image']:
            if inputs.get(var_name) is not None:
                im = inputs[var_name]
                break
        if im is None:
            return None
        if isinstance(im, (list, tuple)):
            im = im[0]
        if isinstance(im, np.ndarray):
            return im.shape[:2]
        if torch.is_tensor(im):
            return (im.shape[-2], im.shape[-1])
        raise Exception('Unknown image type')

    def roll(self):
        return None

    def transform_image(self, image, *rand_params):
        """Must be deterministic"""
        return image


class ToTensor(TransformBase):
    """ Convert to a Tensor"""
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def transform_image(self, image):
        # handle numpy array
        if image.ndim == 2:
            image = image[:, :, None]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # backward compatibility
        if self.normalize:
            return image.float().div(255)
        else:
            return image.float()


class ToTensorAndJitter(TransformBase):
    """ Convert to a Tensor and jitter brightness"""
    def __init__(self, brightness_jitter=0.0, normalize=True):
        super().__init__()
        self.brightness_jitter = brightness_jitter
        self.normalize = normalize

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform_image(self, image, brightness_factor):
        # handle numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))

        # backward compatibility
        if self.normalize:
            return image.float().mul(brightness_factor/255.0).clamp(0.0, 1.0)
        else:
            return image.float().mul(brightness_factor).clamp(0.0, 255.0)


class Normalize(TransformBase):
    """Normalize image"""
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def transform_image(self, image):
        return tvisf.normalize(image, self.mean, self.std, self.inplace)


class ToGrayscale(TransformBase):
    """Converts image to grayscale with probability"""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_grayscale):
        if do_grayscale:
            if torch.is_tensor(image):
                raise NotImplementedError('Implement torch variant.')
            img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return image


class ToBGR(TransformBase):
    """Converts image to grayscale with probability"""
    def transform_image(self, image):
        if torch.is_tensor(image):
            raise NotImplementedError('Implement torch variant.')
        img_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return img_bgr


class RandomHorizontalFlip(TransformBase):
    """Horizontally flip the given NumPy Image randomly with a probability p."""
    def __init__(self, probability = 0.5):
        super().__init__()
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform_image(self, image, do_flip):
        if do_flip:
            if torch.is_tensor(image):
                return image.flip((2,))
            return np.fliplr(image).copy()
        return image
