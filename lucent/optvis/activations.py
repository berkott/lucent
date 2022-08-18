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

from collections import OrderedDict
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import time

from lucent.optvis import objectives
from lucent.optvis import hooks


def get_activation(
    model,
    dataset_img, # As PIL image or directory string
    objective_f,
    verbose=False
):
    model_hook = hooks.ModelHooks(model)
    model_hook.hook_model()
    
    objective_f = objectives.as_objective(objective_f)

    if isinstance(dataset_img, str):
        data_tensor = pil_image_to_tensor(Image.open(dataset_img))

        with torch.no_grad():
            if verbose:
                t0 = time.time()
            model(data_tensor)
            # model_hook.model(data_tensor)
            if verbose:
                t1 = time.time()
                print(f"Total time activation: {t1-t0}")

    elif Image.Image:
        data_tensor = pil_image_to_tensor(dataset_img)

        with torch.no_grad():
            if verbose:
                t0 = time.time()
            model(data_tensor)
            # model_hook.model(data_tensor)
            if verbose:
                t1 = time.time()
                print(f"Total time activation: {t1-t0}")

    else:
        raise TypeError("Can only take string or PIL Image as dataset_img.")
    
    activation = objective_f(model_hook.get_hook())

    if verbose:
        print(objective_f.description)

    model_hook.un_hook_model()

    return activation.item(), objective_f.description

# def get_activation(
#     model,
#     dataset_img, # As PIL image or directory string
#     objective_f,
#     verbose=False
# ):
#     hook = hook_model(model)

#     objective_f = objectives.as_objective(objective_f)

#     if isinstance(dataset_img, str):
#         data_tensor = pil_image_to_tensor(Image.open(dataset_img))

#         with torch.no_grad():
#             if verbose:
#                 t0 = time.time()
#             model(data_tensor)
#             if verbose:
#                 t1 = time.time()
#                 print(f"Total time activation: {t1-t0}")

#     elif Image.Image:
#         data_tensor = pil_image_to_tensor(dataset_img)

#         with torch.no_grad():
#             if verbose:
#                 t0 = time.time()
#             model(data_tensor)
#             if verbose:
#                 t1 = time.time()
#                 print(f"Total time activation: {t1-t0}")

#     else:
#         raise TypeError("Can only take string or PIL Image as dataset_img.")
    
#     activation = objective_f(hook)

#     if verbose:
#         print(objective_f.description)

#     remove_all_forward_hooks(model)

#     return activation.item(), objective_f.description


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
    ])

    # TODO: See if accounting for the batch size like this even makes sense
    return transforms(img)[None, :].to(device)


# class ModuleHook:
#     def __init__(self, module):
#         self.hook = module.register_forward_hook(self.hook_fn)
#         self.module = None
#         self.features = None

#     def hook_fn(self, module, input, output):
#         self.module = module
#         self.features = output

#     def close(self):
#         self.hook.remove()


# def hook_model(model, image_f=None):
#     features = OrderedDict()

#     # recursive hooking function
#     def hook_layers(net, prefix=[]):
#         if hasattr(net, "_modules"):
#             for name, layer in net._modules.items():
#                 if layer is None:
#                     # e.g. GoogLeNet's aux1 and aux2 layers
#                     continue
#                 features["_".join(prefix + [name])] = ModuleHook(layer)
#                 hook_layers(layer, prefix=prefix + [name])

#     hook_layers(model)

#     def hook(layer):
#         if layer == "input":
#             assert image_f is not None, "image_f must be passed into hook_model() if input layer is accessed"
#             out = image_f()
#         elif layer == "labels":
#             out = list(features.values())[-1].features
#         else:
#             assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
#             out = features[layer].features
#         assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
#         return out

#     def get_features():
#         return features

#     return hook, get_features


# def remove_all_forward_hooks(model):
#     for _, child in model._modules.items():
#         if child is not None:
#             if hasattr(child, "_forward_hooks"):
#                 child._forward_hooks = OrderedDict()
#             remove_all_forward_hooks(child)
