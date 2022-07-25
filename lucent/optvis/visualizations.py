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

import numpy as np
import torch
import torch.nn.functional as F
from decorator import decorator
from lucent.optvis.objectives_util import _make_arg_str, _extract_act_pos, _T_handle_batch


def deit_attention_pattern(layer, transformer_input, head_ix, patch_ix, num_attention_heads=6):
    """Encourage diversity between each batch element.

    A neural net feature often responds to multiple things, but naive feature
    visualization often only shows us one. If you optimize a batch of images,
    this objective will encourage them all to be different.

    In particular, it calculates the correlation matrix of activations at layer
    for each image, and then penalizes cosine similarity between them. This is
    very similar to ideas in style transfer, except we're *penalizing* style
    similarity instead of encouraging it.

    Args:
        layer: layer to evaluate activation correlations on.

    Returns:
        Tensor of visualization in right size
    """

    def inner(model):
        layer_t = model(layer)

        # TODO: Maybe fix how transformer input is used here, this is kinda awk
        batches, patches, d_model = model(transformer_input).shape

        qkv = layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)
        attn = attn.softmax(dim=-1)
        
        img_dists = attn[:, head_ix, patch_ix, :]
        patch_dim = round((img_dists.shape[1] - 1)**0.5)

        final_img = torch.zeros(1, 3, patch_dim, patch_dim)
        final_img[0, 0] = img_dists[0, 1:].reshape(patch_dim, patch_dim)
        final_img[0, 1] = torch.full((patch_dim, patch_dim), img_dists[0, 0].item())
        
        return final_img
        # return [dist[1:].reshape(1, round((dist.shape[0] - 1)**0.5), round((dist.shape[0] - 1)**0.5)) for dist in img_dists]

    return inner

# class Visualizations():

#     def __init__(self, objective_func, name="", description=""):
#         self.objective_func = objective_func
#         self.name = name
#         self.description = description

#     def __call__(self, model):
#         return self.objective_func(model)

#     def __add__(self, other):
#         if isinstance(other, (int, float)):
#             objective_func = lambda model: other + self(model)
#             name = self.name
#             description = self.description
#         else:
#             objective_func = lambda model: self(model) + other(model)
#             name = ", ".join([self.name, other.name])
#             description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
#         return Objective(objective_func, name=name, description=description)

#     @staticmethod
#     def sum(objs):
#         objective_func = lambda T: sum([obj(T) for obj in objs])
#         descriptions = [obj.description for obj in objs]
#         description = "Sum(" + " +\n".join(descriptions) + ")"
#         names = [obj.name for obj in objs]
#         name = ", ".join(names)
#         return Objective(objective_func, name=name, description=description)

#     def __neg__(self):
#         return -1 * self

#     def __sub__(self, other):
#         return self + (-1 * other)

#     def __mul__(self, other):
#         if isinstance(other, (int, float)):
#             objective_func = lambda model: other * self(model)
#             return Objective(objective_func, name=self.name, description=self.description)
#         else:
#             # Note: In original Lucid library, objectives can be multiplied with non-numbers
#             # Removing for now until we find a good use case
#             raise TypeError('Can only multiply by int or float. Received type ' + str(type(other)))

#     def __truediv__(self, other):
#         if isinstance(other, (int, float)):
#             return self.__mul__(1 / other)
#         else:
#             raise TypeError('Can only divide by int or float. Received type ' + str(type(other)))

#     def __rmul__(self, other):
#         return self.__mul__(other)

#     def __radd__(self, other):
#         return self.__add__(other)


# def wrap_objective():
#     @decorator
#     def inner(func, *args, **kwds):
#         objective_func = func(*args, **kwds)
#         objective_name = func.__name__
#         args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
#         description = objective_name.title() + args_str
#         return Objective(objective_func, objective_name, description)
#     return inner


# def handle_batch(batch=None):
#     return lambda f: lambda model: f(_T_handle_batch(model, batch=batch))

# @wrap_objective()
# def deit_attention_pattern(layer, transformer_input, head_ix, patch_ix, num_attention_heads=6):
#     """Encourage diversity between each batch element.

#     A neural net feature often responds to multiple things, but naive feature
#     visualization often only shows us one. If you optimize a batch of images,
#     this objective will encourage them all to be different.

#     In particular, it calculates the correlation matrix of activations at layer
#     for each image, and then penalizes cosine similarity between them. This is
#     very similar to ideas in style transfer, except we're *penalizing* style
#     similarity instead of encouraging it.

#     Args:
#         layer: layer to evaluate activation correlations on.

#     Returns:
#         Tensor of visualization in right size
#     """
#     # def get_map_from_dist(dist, img_size):
#     #   class_embedding = dist[0]
#     #   dist = dist[1:]
#     #   img_patches = dist.shape[0]
#     #   img_patch_size = round(((img_size**2) // img_patches)**.5)

#     #   img = dist.view(1, img_patches, 1)
#     #   img = img.expand(1, img_patches, img_patch_size)
#     #   img = img.reshape(1, img_patch_size, img_patches)
#     #   img = img.tile((1, img_patch_size))
#     #   return img.view(1, img_size, img_size)

#     # def inner(model, img_size):
#     #     layer_t = model(layer)

#     #     # TODO: Maybe fix how transformer input is used here, this is kinda awk
#     #     batches, patches, d_model = model(transformer_input).shape

#     #     qkv = layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
#     #     q, k = qkv[0], qkv[1]

#     #     attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)

#     #     attn = attn.softmax(dim=-1)
        
#     #     img_dists = attn[:, head_ix, patch_ix, :]

#     #     return [get_map_from_dist(dist, img_size) for dist in img_dists]


#     def inner(model):
#         layer_t = model(layer)

#         # TODO: Maybe fix how transformer input is used here, this is kinda awk
#         batches, patches, d_model = model(transformer_input).shape

#         qkv = layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
#         q, k = qkv[0], qkv[1]

#         attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)
#         attn = attn.softmax(dim=-1)
        
#         img_dists = attn[:, head_ix, patch_ix, :]
#         return [dist[1:].reshape(1, dist.shape[0] - 1, dist.shape[0] - 1) for dist in img_dists]

#     return inner


# def as_objective(obj):
#     """Convert obj into Objective class.

#     Strings of the form "layer:n" become the Objective channel(layer, n).
#     Objectives are returned unchanged.

#     Args:
#         obj: string or Objective.

#     Returns:
#         Objective
#     """
#     if isinstance(obj, Objective):
#         return obj
#     if callable(obj):
#         return obj
#     if isinstance(obj, str):
#         layer, chn = obj.split(":")
#         layer, chn = layer.strip(), int(chn)
#         return channel(layer, chn)
