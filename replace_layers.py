"""Replace layers in a Pytorch model with desired custom layers.
"""
import warnings
from transformers.models.opt.modeling_opt import OPTAttention
from softmax_free_attention import SoftmaxFreeOptAttention

DEFAULT_MAPPING = {
    OPTAttention: SoftmaxFreeOptAttention
}


def replace_layers(module, mapping=None, num_blocks=2, seqlen=1024):
    """Replace torch.nn.Module layers with custom layers.

    Arguments:
        module (torch.nn.Module): The model to be converted. This model is
            modified in-place.
        mapping (dict): a mapping from nn.Module to custom nn.Module layers.
        num_blocks (int): Number of chunks to split the sequence length dim into.
        seqlen (int): The length of the sequence.
    """
    mapping = mapping or DEFAULT_MAPPING

    def swap(module, mapping, replace_func="from_float"):
        # To support more general replacement patterns, we allow the user
        # to override which function is called - from_float is the default.
        target = mapping[type(module)]
        try:
            replaced = getattr(target, replace_func)(module, num_blocks, seqlen)
        # If some attributes don't match between base and custom modules.
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(
                f"{target}.from_float({module}) failed with:\n\t{e}"
            )
        return replaced

    def convert(module, mapping):
        key = type(module)
        if key in mapping:
            return swap(module, mapping)
        else:
            named_children = list(module.named_children())
            if len(named_children):
                reassign = {}
                for name, child in module.named_children():
                    reassign[name] = convert(child, mapping)

                for key, value in reassign.items():
                    module._modules[key] = value
            else:
                # Print message if a non-Pytorch module is encountered.
                python_module_name = key.__module__
                if not python_module_name.startswith("torch.nn"):
                    msg = (
                        f"Encountered a non-Pytorch module: {python_module_name}"
                    )
                    warnings.warn(msg)

            return module

    return convert(module, mapping)
