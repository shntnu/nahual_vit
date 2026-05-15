from functools import partial
from typing import Callable

import torch
from transformers import AutoModel

# We will use pre-existing information to enforce guardrails on the input data
# model -> (expected #channels, mandated shape of yx)
guardrail_shapes = {
    "recursionpharma/OpenPhenom": (None, 16),
    "CaicedoLab/MorphEm": (None, 16),
}


def _resolve_device(device) -> torch.device:
    """Map a client-supplied device hint to a torch.device the host supports.

    Strings ("cuda:0", "mps", "cpu") pass through. Ints select CUDA when
    available, fall back to MPS on Apple Silicon, else CPU - so a notebook
    that hard-codes `device=2` still launches on a Mac.
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", int(device))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_base(model_name: str, process_pixels: Callable, **kwargs) -> dict:
    # Some default values
    device = kwargs.get("device", 0)

    setup_defaults = dict(
        trust_remote_code=True,
        dtype="auto",
        device=_resolve_device(device),
    )
    execution_defaults = dict()

    setup_kwargs = kwargs.get("setup_kwargs", {})
    execution_kwargs = kwargs.get("setup_kwargs", {})

    # Define parameters by combining defaults and non-defaults
    setup_params = {**setup_defaults, **setup_kwargs}

    # Device is not in from_pretrained
    device = setup_params.pop("device")

    execution_params = {**execution_defaults, **execution_kwargs}

    # Load model instance
    model = AutoModel.from_pretrained(model_name, **setup_params).to(device)
    model.eval()

    expected_channels, yx_shape = guardrail_shapes[model_name]
    execution_params["expected_channels"] = expected_channels
    execution_params["expected_yx"] = yx_shape

    # Generate a json-encodable dictionary to send back to the client
    serializable_params = {
        name: {k: str(v) for k, v in d.items()}
        for name, d in zip(("setup", "execution"), (setup_params, execution_params))
    }

    # "Freeze" model in-place
    processor = partial(process_pixels, model=model, device=device, **execution_params)
    return processor, serializable_params
