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

# from collections import OrderedDict
import time
import os
# import json
# import pickle

from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from lucent.optvis import objectives
from lucent.optvis import hooks


def get_activation(
    model,
    dataset_img, # As PIL image or directory string
    verbose=False
):
    model_hook = hooks.ModelHooks(model)
    model_hook.hook_model()

    if isinstance(dataset_img, str):
        data_tensor = pil_image_to_tensor(Image.open(dataset_img))
    elif Image.Image:
        data_tensor = pil_image_to_tensor(dataset_img)
    else:
        raise TypeError("Can only take string or PIL Image as dataset_img.")

    with torch.no_grad():
        if verbose:
            t0 = time.time()
        model(data_tensor)
        # model_hook.model(data_tensor)
        if verbose:
            t1 = time.time()
            print(f"Total time activation: {t1-t0}")

    model_activations = model_hook.get_hook()
    model_hook.un_hook_model()

    def inner_get_activation(objective_f):
        if isinstance(objective_f, objectives.Objective):
            objective_f = objectives.as_objective(objective_f)
            activation = objective_f(model_activations)

            if verbose:
                print(objective_f.description)

            return activation.item(), objective_f.description
        else:
            return objective_f(model_activations)

    return inner_get_activation


# def get_activation(
#     model,
#     dataset_img, # As PIL image or directory string
#     objective_f,
#     verbose=False
# ):
#     model_hook = hooks.ModelHooks(model)
#     model_hook.hook_model()
    
#     objective_f = objectives.as_objective(objective_f)

#     if isinstance(dataset_img, str):
#         data_tensor = pil_image_to_tensor(Image.open(dataset_img))
#     elif Image.Image:
#         data_tensor = pil_image_to_tensor(dataset_img)
#     else:
#         raise TypeError("Can only take string or PIL Image as dataset_img.")

#     with torch.no_grad():
#         if verbose:
#             t0 = time.time()
#         model(data_tensor)
#         # model_hook.model(data_tensor)
#         if verbose:
#             t1 = time.time()
#             print(f"Total time activation: {t1-t0}")
    
#     activation = objective_f(model_hook.get_hook())

#     if verbose:
#         print(objective_f.description)

#     model_hook.un_hook_model()

#     return activation.item(), objective_f.description


# def save_activations(
#     model,
#     dataset_img, # As PIL image or directory string
#     save_path,
#     verbose=False
# ):
#     model_hook = hooks.ModelHooks(model)
#     model_hook.hook_model()
    
#     if isinstance(dataset_img, str):
#         data_tensor = pil_image_to_tensor(Image.open(dataset_img))
#     elif Image.Image:
#         data_tensor = pil_image_to_tensor(dataset_img)
#     else:
#         raise TypeError("Can only take string or PIL Image as dataset_img.")

#     with torch.no_grad():
#         if verbose:
#             t0 = time.time()
#         model(data_tensor)
#         if verbose:
#             t1 = time.time()
#             print(f"Total time activation: {t1-t0}")

#     full_activations = model_hook.get_features()
#     export_file(full_activations, save_path)

#     model_hook.un_hook_model()

#     return full_activations


# def export_file(data, save_path): 
#     try:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     except Exception as e:
#         print(e)

#     pickle.dump(data, open(save_path, "wb"))

# def import_file(save_path): 
#     try:
#         return pickle.load(open(save_path, "rb"))
#     except Exception as e:
#         print("Could not find file")
#         print(e)

def pil_image_to_tensor(img, img_size=224):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    transforms = Compose([
        Resize(img_size),
        CenterCrop(img_size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TODO: See if accounting for the batch size like this even makes sense
    return transforms(img)[None, :].to(device)
