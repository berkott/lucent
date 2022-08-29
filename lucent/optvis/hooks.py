from collections import OrderedDict

class ModelHooks:
    def __init__(self, model, image_f=None):
        self.model = model
        self.features = OrderedDict()
        self.image_f = image_f

    def hook_model(self):
        self.__hook_model_rec(self.model)

    # recursive hooking function
    def __hook_model_rec(self, model, prefix=[]):
        if hasattr(model, "_modules"):
            for name, layer in model._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                self.features["_".join(prefix + [name])] = self.ModuleHook(layer)
                self.__hook_model_rec(layer, prefix=prefix + [name])

    def un_hook_model(self):
        self.__un_hook_model_rec(self.model)

    def __un_hook_model_rec(self, model):
        for name, child in model._modules.items():
            if child is not None:
                if hasattr(child, "_forward_hooks"):
                    child._forward_hooks = OrderedDict()
                self.__un_hook_model_rec(child)

    def get_hook(self):
        def hook(layer):
            if layer == "input":
                assert self.image_f is not None, "image_f must be passed into hook_model() if input layer is accessed"
                out = self.image_f()
            elif layer == "labels":
                out = list(self.features.values())[-1].features
            else:
                assert layer in self.features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
                out = self.features[layer].features
            # assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example." TODO: See if I can uncomment this line??
            return out

        return hook

    def get_features(self):
        return self.features
    
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
