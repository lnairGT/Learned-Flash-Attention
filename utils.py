import warnings


def assign_param(module, other_module, attr_name):
    param = getattr(module, attr_name)
    if param is None:
        return
    other_param = getattr(other_module, attr_name)

    module_dtype = param.data.dtype
    other_module_dtype = other_param.data.dtype

    if module_dtype != other_module_dtype:
        warnings.warn(
            f"Casting {attr_name} from {module_dtype} to "
            f"{other_module_dtype}. Training optimization may fail "
            "depending on the order in which the optimizer class is "
            "initialized."
        )
        param.data = other_param.data.to(module.backend.dtype)
    else:
        # These are raw torch.nn.Parameters. Training with this layer
        # will work regardless of whether the optimizer was given the
        # original module's parameters or this module's parameters.
        setattr(module, attr_name, getattr(other_module, attr_name))
