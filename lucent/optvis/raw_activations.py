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
import torch


def full_residual_stream(layer, patch_ix):
    def inner(model):
        return model(layer)[0, patch_ix, :]
    return inner


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

def attention_pattern_finder(patch_ix, total_patches):
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

    def inner(_):
        patch_dim = round((total_patches - 1)**0.5)
        final_img = torch.zeros(1, 3, patch_dim, patch_dim)
        
        color_channel = torch.zeros(total_patches - 1)
        color_channel[max(patch_ix - 1, 0)] = 1

        final_img[0, 1] = color_channel.reshape(patch_dim, patch_dim)
        final_img[0, 2] = color_channel.reshape(patch_dim, patch_dim)
        return final_img

    return inner
