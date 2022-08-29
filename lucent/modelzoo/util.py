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

"""Utility functions for modelzoo models."""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from lucent.optvis import hooks
import torch


def get_model_layers(model, getLayerRepr=False, getLayerDims=None):
    """
    If getLayerRepr is True, return a OrderedDict of layer names, layer representation string pair.
    If it's False, just return a list of layer names
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layers = OrderedDict() if getLayerRepr or getLayerDims is not None else []

    if getLayerDims is not None:
        model_hook = hooks.ModelHooks(model)
        model_hook.hook_model()
        
        with torch.no_grad():
            model(torch.zeros(getLayerDims).to(device))

        model_activations = model_hook.get_hook()
        model_hook.un_hook_model()

    
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                concat_name = "_".join(prefix + [name])

                model_activation_size = model_activations(concat_name).size() if model_activations(concat_name) is not None else None

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if getLayerRepr and not getLayerDims:
                    layers[concat_name] = layer.__repr__()
                elif not getLayerRepr and getLayerDims is not None:
                    # layers[concat_name] = model_activations(concat_name).size()
                    layers[concat_name] = model_activation_size
                elif getLayerRepr and getLayerDims is not None:
                    layers[concat_name] = [layer.__repr__(), model_activation_size]
                else:
                    layers.append(concat_name)
                get_layers(layer, prefix=prefix+[name])

    get_layers(model)
    return layers
