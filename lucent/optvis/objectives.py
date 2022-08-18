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


class Objective():
    
    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, model, *args):
        # if model_2:
        #     print(model_2)
        #     print(self.objective_func)
        #     print(self.objective_func)
        #     import inspect
  
        #     print(inspect.getargspec(self.objective_func))
        # return self.objective_func(model, model_2) if model_2 else self.objective_func(model)

        try:
            return self.objective_func(model, *args)
        except TypeError:
            return self.objective_func(model)

        # return self.objective_func(model, *args)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model, *args: other + self(model, *args)
            try:
                objective_func = lambda model, *args: other + self(model, *args)
            except TypeError:
                objective_func = lambda model: other + self(model)
            name = self.name
            description = self.description
        else:
            try:
                objective_func = lambda model, *args: self(model, *args) + other(model, *args)
            except TypeError:
                try:
                    objective_func = lambda model, *args: self(model, *args) + other(model)
                except TypeError:
                    objective_func = lambda model, *args: self(model) + other(model, *args)
            
            name = ", ".join([self.name, other.name])
            description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
        return Objective(objective_func, name=name, description=description)

    @staticmethod
    def sum(objs):
        # objective_func = lambda T, *args: sum([obj(T, *args) for obj in objs])
        # TODO: Do I need the *args?
        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names)
        return Objective(objective_func, name=name, description=description)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            try:
                objective_func = lambda model, *args: other * self(model, *args)
            except TypeError:
                objective_func = lambda model: other * self(model)
            return Objective(objective_func, name=self.name, description=self.description)
        else:
            # Note: In original Lucid library, objectives can be multiplied with non-numbers
            # Removing for now until we find a good use case
            raise TypeError('Can only multiply by int or float. Received type ' + str(type(other)))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1 / other)
        else:
            raise TypeError('Can only divide by int or float. Received type ' + str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)


def wrap_objective():
    @decorator
    def inner(func, *args, **kwds):
        objective_func = func(*args, **kwds)
        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        return Objective(objective_func, objective_name, description)
    return inner


# TODO: add *args to this func
def handle_batch(batch=None):
    return lambda f: lambda model: f(_T_handle_batch(model, batch=batch))


@wrap_objective()
def neuron(layer, n_channel, x=None, y=None, batch=None):
    """Visualize a single neuron of a single channel.

    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.

    Odd width & height:               Even width & height:

    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+

    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -layer_t[:, n_channel].mean()
    return inner


@wrap_objective()
def channel(layer, n_channel, batch=None):
    """Visualize a single channel"""
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, n_channel].mean()
    return inner

@wrap_objective()
def neuron_weight(layer, weight, x=None, y=None, batch=None):
    """ Linearly weighted channel activation at one location as objective
    weight: a torch Tensor vector same length as channel.
    """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        if weight is None:
            return -layer_t.mean()
        else:
            return -(layer_t.squeeze() * weight).mean()
    return inner

@wrap_objective()
def channel_weight(layer, weight, batch=None):
    """ Linearly weighted channel activation as objective
    weight: a torch Tensor vector same length as channel. """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        return -(layer_t * weight.view(1, -1, 1, 1)).mean()
    return inner

@wrap_objective()
def localgroup_weight(layer, weight=None, x=None, y=None, wx=1, wy=1, batch=None):
    """ Linearly weighted channel activation around some spot as objective
    weight: a torch Tensor vector same length as channel. """
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        if weight is None:
            return -(layer_t[:, :, y:y + wy, x:x + wx]).mean()
        else:
            return -(layer_t[:, :, y:y + wy, x:x + wx] * weight.view(1, -1, 1, 1)).mean()
    return inner

@wrap_objective()
def direction(layer, direction, batch=None):
    """Visualize a direction

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """
    @handle_batch(batch)
    def inner(model):
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), model(layer)).mean()

    return inner


@wrap_objective()
def direction_neuron(layer,
                     direction,
                     x=None,
                     y=None,
                     batch=None):
    """Visualize a single (x, y) position along the given direction

    Similar to the neuron objective, defaults to the center neuron.

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction_neuron(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)

    Returns:
        Objective

    """

    @handle_batch(batch)
    def inner(model):
        # breakpoint()
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -torch.nn.CosineSimilarity(dim=1)(direction.reshape(
            (1, -1, 1, 1)), layer_t).mean()

    return inner


def _torch_blur(tensor, out_c=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth = tensor.shape[1]
    weight = np.zeros([depth, depth, out_c, out_c])
    for ch in range(depth):
        weight_ch = weight[ch, ch, :, :]
        weight_ch[ :  ,  :  ] = 0.5
        weight_ch[1:-1, 1:-1] = 1.0
    weight_t = torch.tensor(weight).float().to(device)
    conv_f = lambda t: F.conv2d(t, weight_t, None, 1, 1)
    return conv_f(tensor) / conv_f(torch.ones_like(tensor))


@wrap_objective()
def blur_input_each_step():
    """Minimizing this objective is equivelant to blurring input each step.
    Optimizing (-k)*blur_input_each_step() is equivelant to:
    input <- (1-k)*input + k*blur(input)
    An operation that was used in early feature visualization work.
    See Nguyen, et al., 2015.
    """
    def inner(T):
        t_input = T("input")
        with torch.no_grad():
            t_input_blurred = _torch_blur(t_input)
        return -0.5*torch.sum((t_input - t_input_blurred)**2)
    return inner


@wrap_objective()
def channel_interpolate(layer1, n_channel1, layer2, n_channel2):
    """Interpolate between layer1, n_channel1 and layer2, n_channel2.
    Optimize for a convex combination of layer1, n_channel1 and
    layer2, n_channel2, transitioning across the batch.
    Args:
        layer1: layer to optimize 100% at batch=0.
        n_channel1: neuron index to optimize 100% at batch=0.
        layer2: layer to optimize 100% at batch=N.
        n_channel2: neuron index to optimize 100% at batch=N.
    Returns:
        Objective
    """
    def inner(model):
        batch_n = list(model(layer1).shape)[0]
        arr1 = model(layer1)[:, n_channel1]
        arr2 = model(layer2)[:, n_channel2]
        weights = np.arange(batch_n) / (batch_n - 1)
        sum_loss = 0
        for n in range(batch_n):
            sum_loss -= (1 - weights[n]) * arr1[n].mean()
            sum_loss -= weights[n] * arr2[n].mean()
        return sum_loss
    return inner


@wrap_objective()
def alignment(layer, decay_ratio=2):
    """Encourage neighboring images to be similar.
    When visualizing the interpolation between two objectives, it's often
    desirable to encourage analogous objects to be drawn in the same position,
    to make them more comparable.
    This term penalizes L2 distance between neighboring images, as evaluated at
    layer.
    In general, we find this most effective if used with a parameterization that
    shares across the batch. (In fact, that works quite well by itself, so this
    function may just be obsolete.)
    Args:
        layer: layer to penalize at.
        decay_ratio: how much to decay penalty as images move apart in batch.
    Returns:
        Objective.
    """
    def inner(model):
        batch_n = list(model(layer).shape)[0]
        layer_t = model(layer)
        accum = 0
        for d in [1, 2, 3, 4]:
            for i in range(batch_n - d):
                a, b = i, i + d
                arr_a, arr_b = layer_t[a], layer_t[b]
                accum += ((arr_a - arr_b) ** 2).mean() / decay_ratio ** float(d)
        return accum
    return inner


@wrap_objective()
def diversity(layer):
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
        Objective.
    """
    def inner(model):
        layer_t = model(layer)
        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return -sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch
    return inner


@wrap_objective()
def L2(layer="input", batch=None):
    """L2 norm of layer. Generally used as penalty."""
    @handle_batch(batch)
    def inner(model):
        # print(model(layer).shape)
        # print(torch.linalg.norm(model(layer)))
        return torch.linalg.norm(model(layer))
    return inner


@wrap_objective()
def custom_objective(custom_objective_func, batch=None):
    """Write a custom objective function given the model layer"""
    return handle_batch(batch)(custom_objective_func)


# ViT and MLP-Mixer objectives
@wrap_objective()
def neuron_two_index(layer, patch_ix, neuron_ix, batch=None):
    """Write a custom objective function given the model layer"""
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[patch_ix, :, neuron_ix].mean()
    return inner


@wrap_objective()
def residual_stream(layer, patch_ix, neuron_ix, batch=None):
    """Write a custom objective function given the model layer"""
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, patch_ix, neuron_ix].mean()
    return inner


@wrap_objective()
def patch(layer, neuron_ix, batch=None):
    """Write a custom objective function given the model layer"""
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, :, neuron_ix].mean()
    return inner


@wrap_objective()
def layer_diversity(layer):
    """Encourage diversity between each batch element.

    A neural net feature often responds to multiple things, but naive feature
    visualization often only shows us one. If you optimize a batch of images,
    this objective will encourage them all to be different.

    In particular, it calculates an approximation of the correlation matrix of 
    activations at layer, namely the gram matrix, for each image, and then 
    penalizes cosine similarity between them. This is very similar to ideas 
    in style transfer, except we're *penalizing* style similarity instead of 
    encouraging it.

    Args:
        layer: layer to evaluate activation correlations on.

    Returns:
        Objective.
    """
    def inner(model):
        layer_t = model(layer)
        patches, batch, _ = layer_t.shape
        flattened = layer_t.view(batch, patches, -1)

        # Wow this is the gram matrix: https://en.wikipedia.org/wiki/Gram_matrix
        # grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))/1000
        # print(grams)
        
        # print(torch.linalg.norm(grams))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        # grams = torch.linarg.matrix_norm(grams)

        # print(grams)

        # print(-sum([ sum([ (grams[i]*grams[j]).sum()
        #        for j in range(batch) if j != i])
        #        for i in range(batch)]) / batch)

        return -sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch
    return inner

@wrap_objective()
def patch_diversity(layer, ix1):
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
        Objective.
    """
    def inner(model):
        patch_t = model(layer)[ix1]
        batch, _ = patch_t.shape
        flattened = patch_t.view(batch, -1)
        # correlation_matrix = torch.corrcoef(flattened)
        # return -torch.sum(correlation_matrix)


        # print(flattened)
        # print(F.cosine_similarity(flattened, torch.roll(flattened, 1, 0)))
        # print(torch.roll(flattened, 1, 0))
        # return F.cosine_similarity(flattened, torch.roll(flattened, 1, 0)).sum()

        # print(flattened.shape)
        # sin = sqrt (1 - (cos theta)^2), this might be better for diversity because it is 0 when they are orthogonal, so very different, but 1 if they are very similar 
        # grams = torch.matmul(flattened, flattened.T)
        # print(grams.shape)
        grams = F.normalize(flattened, p=2, dim=1)
        # print(grams.shape)
        return -sum([sum([(grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch
    return inner


@wrap_objective()
def neuron_two_index_diversity(layer, ix1, ix2):
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
        Objective.
    """
    def inner(model):
        layer_t = model(layer)
        # model(layer)[ix1][0][ix2]
        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return -sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch
    return inner


@wrap_objective()
def attention_pre_softmax_neuron(layer, transformer_input, head_ix, q_ix, k_ix, num_attention_heads=6):
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
        Objective.
    """

    def inner(model):
        layer_t = model(layer)
        # print(model("blocks_0_norm1"))
        # print(model("blocks_0_norm1").shape)

        # TODO: Maybe fix how transformer input is used here, this is kinda awk
        batches, patches, d_model = model(transformer_input).shape

        qkv = layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        # 1, num_heads, num_patches, num_patches
        attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)
                
        return attn[:, head_ix, q_ix, k_ix].mean()
    return inner


@wrap_objective()
def attention_post_softmax_neuron(layer, transformer_input, head_ix, q_ix, k_ix, num_attention_heads=6):
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
        Objective.
    """

    def inner(model):
        layer_t = model(layer)
        # print(model("blocks_0_norm1"))
        # print(model("blocks_0_norm1").shape)

        # TODO: Maybe fix how transformer input is used here, this is kinda awk
        batches, patches, d_model = model(transformer_input).shape

        qkv = layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        # 1, num_heads, num_patches, num_patches
        attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)
        attn = attn.softmax(dim=-1)
                
        return attn[:, head_ix, q_ix, k_ix].mean()
    return inner


@wrap_objective()
def attention_pre_softmax_max_key(layer, transformer_input, head_ix, q_ix, num_attention_heads=6):
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
        Objective.
    """

    def inner(model):
        layer_t = model(layer)
        # print(model("blocks_0_norm1"))
        # print(model("blocks_0_norm1").shape)

        # TODO: Maybe fix how transformer input is used here, this is kinda awk
        batches, patches, d_model = model(transformer_input).shape

        qkv = layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        # 1, num_heads, num_patches, num_patches
        attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)
        
        # return attn[:, head_ix, q_ix, :][0].argmax()
        return torch.argmax(attn[:, head_ix, q_ix, :][0])
    return inner


@wrap_objective()
def attention_pre_softmax_qk_match_patch(layer, transformer_input, head_ix, num_attention_heads=6):
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
        Objective.
    """

    def inner(model):
        layer_t = model(layer)
        # print(model("blocks_0_norm1"))
        # print(model("blocks_0_norm1").shape)

        # TODO: Maybe fix how transformer input is used here, this is kinda awk
        batches, patches, d_model = model(transformer_input).shape

        qkv = layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        # 1, num_heads, num_patches, num_patches
        attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)
        
        return torch.diagonal(attn[:, head_ix, :, :], dim1=1, dim2=2).mean()
    return inner

@wrap_objective()
def attention_pre_softmax_neuron_given_query(layer, transformer_input, head_ix, patch_ix, softmax_patch_ix, num_attention_heads=6):
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
        Objective.
    """

    def inner(model_k, model_q):
        # TODO: Maybe fix how transformer input is used here, this is kinda awk
        batches, patches, d_model = model_q(transformer_input).shape # These dims will be the same because the models are just copies of each other

        q_layer_t = model_q(layer)
        qkv = q_layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]

        k_layer_t = model_k(layer)
        qkv = k_layer_t.reshape(batches, patches, 3, num_attention_heads, d_model // num_attention_heads).permute(2, 0, 3, 1, 4)
        k = qkv[1]

        attn = (q @ k.transpose(-2, -1)) * (d_model // num_attention_heads)
        
        return attn[:, head_ix, patch_ix, softmax_patch_ix].mean()
    return inner


def as_objective(obj):
    """Convert obj into Objective class.

    Strings of the form "layer:n" become the Objective channel(layer, n).
    Objectives are returned unchanged.

    Args:
        obj: string or Objective.

    Returns:
        Objective
    """
    if isinstance(obj, Objective):
        return obj
    if callable(obj):
        return obj
    if isinstance(obj, str):
        layer, chn = obj.split(":")
        layer, chn = layer.strip(), int(chn)
        return channel(layer, chn)
