# Copyright 2020 The Lucent Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import, division, print_function

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.nn.modules.upsampling import Upsample # TODO: Make all upsamples identical
import os
import types
import pickle

from lucent.optvis import objectives, transform, param
from lucent.misc.io import show


def render_vis(
    model,
    objective_f,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(512,),
    verbose=False,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None
):
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            def closure():
                optimizer.zero_grad()
                try:
                    model(transform_f(image_f()))
                except RuntimeError as ex:
                    if i == 1:
                        # Only display the warning message
                        # on the first iteration, no need to do that
                        # every iteration
                        warnings.warn(
                            "Some layers could not be computed because the size of the "
                            "image is not big enough. It is fine, as long as the non"
                            "computed layers are not used in the objective function"
                            f"(exception details: '{ex}')"
                        )
                loss = objective_f(hook)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                if verbose:
                    print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                    if show_inline:
                        show(image)
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return images


def render_qk(
    model,
    objective_f,
    input_image_pil,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(512,),
    verbose=False,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None
):
    # One option that could work is creating 2 models and that way the hooks still work and the rest of the code isn't damaged, not elegant tho
    # LMAO im doing this ^, it's kinda bad but idk what else to do ngl
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_q = pickle.loads(pickle.dumps(model)).to(device).eval() # Basically just copies model, deepcopy doesn't work TODO: find more elegant solution
    hook_q = hook_model(model_q)

    hook = hook_model(model, image_f)

    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        model_q(pil_image_to_tensor(input_image_pil))
        
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            def closure():
                optimizer.zero_grad()
                try:
                    model(transform_f(image_f()))
                except RuntimeError as ex:
                    if i == 1:
                        # Only display the warning message
                        # on the first iteration, no need to do that
                        # every iteration
                        warnings.warn(
                            "Some layers could not be computed because the size of the "
                            "image is not big enough. It is fine, as long as the non"
                            "computed layers are not used in the objective function"
                            f"(exception details: '{ex}')"
                        )
                loss = objective_f(hook, hook_q)
                
                loss.backward(retain_graph=True)
                return loss
                
            optimizer.step(closure)
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                if verbose:
                    print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                    if show_inline:
                        show(image)
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return images


def render_attn_pattern(
    model,
    visualization,
    img_array=None,
    img_size=224,
    # verbose=False,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
):
    try:
        hook = hook_model(model)
        model(img_array_to_tensor(img_array))
        tensor_visualization = visualization(hook)
    except:
        try:
            tensor_visualization = visualization()
        except Exception as e:
            print("You must either hook the model and give a valid input image or call a visualization function that doesn't require a model hook.")
            print(e)

    tensor_visualization = Upsample(scale_factor=img_size//tensor_visualization.shape[-1], mode='nearest')(tensor_visualization)

    if save_image:
        export(tensor_visualization, image_name)
    if show_inline:
        show(tensor_to_img_array(tensor_visualization))
    elif show_image:
        view(tensor_visualization)
    return tensor_visualization


def render_data_aug(
    input_image_pil, 
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False
):
    tensor_visualization = pil_image_to_tensor(input_image_pil)
    
    if save_image:
        export(tensor_visualization, image_name)
    if show_inline:
        show(tensor_to_img_array(tensor_visualization))
    elif show_image:
        view(tensor_visualization)
    return tensor_to_img_array(tensor_visualization)


def pil_image_to_tensor(img, img_size=224):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    # transforms = Compose([
    #     Resize(img_size),
    #     CenterCrop(img_size),
    #     _convert_image_to_rgb,
    #     ToTensor(),
    #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) # TODO: is this a principled thing to do? I just copy pasted from CLIP code
    # ])

    transforms = Compose([
        Resize(img_size),
        CenterCrop(img_size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TODO: See if accounting for the batch size like this even makes sense
    return transforms(img)[None, :].to(device).requires_grad_(True)


def img_array_to_tensor(img_array):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_array = np.transpose(img_array, [0, 3, 1, 2])
    tensor = torch.from_numpy(img_array).to(device).requires_grad_(True)
    return tensor


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, image_name=None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)

    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    
    try:
        os.makedirs(os.path.dirname(image_name), exist_ok=True)
    except Exception as e:
        print(e)

    Image.fromarray(image).save(image_name)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, image_f=None):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            assert image_f is not None, "image_f must be passed into hook_model() if input layer is accessed"
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook
