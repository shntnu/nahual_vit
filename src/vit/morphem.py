"""Adapted from https://huggingface.co/CaicedoLab/MorphEm"""

import sys
from functools import partial

import numpy
import pynng
import torch
import torch.nn as nn
import transformers
import trio
from loguru import logger
from nahual.preprocess import validate_input_shape
from nahual.server import responder
from torchvision import transforms as v2

from vit.setup import setup_base


# Pass the setup base to override process_pixels function
def setup(*args, **kwargs):
    return partial(setup_base, process_pixels=process_pixels)(*args, **kwargs)


async def main():
    """Main function for the asynchronous server.

    This function sets up a nng connection using pynng and starts a nursery to handle
    incoming requests asynchronously.

    Parameters
    ----------
    address : str
        The network address to listen on.

    Returns
    -------
    None
    """

    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"Pretrained ViT server listening on {address}")
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


def transform(pixels: numpy.ndarray):
    transform_fn = v2.Compose([
        SaturationNoiseInjector(),
        PerImageNormalize(),
        v2.Resize(size=(224, 224), antialias=True),
    ])
    return transform_fn(pixels)


def process_pixels(
    pixels: numpy.ndarray,
    model: transformers.modeling_utils.PreTrainedModel,
    expected_yx: tuple[int],
    expected_channels: int,
    device: torch.device,
) -> numpy.ndarray:
    """Apply a pretrained model. We pass arguments that encode the necessary input shapes and number of channels to pad. We will valudate the yx dimensions and pad the channel dimension with zeros.

    Input contract (caller side)
    ----------------------------
    pixels : NCZYX uint8 in [0, 255]; H, W divisible by 16; Z=1.
        Channels are processed independently (bag-of-channels) — pass each
        biological channel as a separate slot.

    Server-side normalization (per channel, applied here)
    -----------------------------------------------------
    SaturationNoiseInjector (saturated pixels == 255 → uniform [200, 255])
        → PerImageNormalize (InstanceNorm2d, per-image mean/std)
        → Resize to 224×224 antialias.

    The uint8 dtype is load-bearing: the saturation injector keys on the
    exact value 255, which only survives if the caller passes raw 8-bit
    pixels (not rescaled to [0, 1]).

    Output
    ------
    (N, C × 384) — per-channel CLS tokens concatenated along feature axis.
    """

    _, input_channels, _, *input_yx = pixels.shape

    validate_input_shape(input_yx, expected_yx)

    # pixels = pad_channel_dim(pixels, expected_channels)
    # No padding necessary for morphem, just index z-stack
    pixels = pixels[:, :, 0, :, :]
    n_channels = pixels.shape[1]

    with torch.no_grad():
        batch_feat = []
        pixels_torch = torch.from_numpy(pixels).float().to(device)

        for c in range(n_channels):
            # Extract single channel: (N, C, H, W) -> (N, 1, H, W)
            single_channel = pixels_torch[:, c, :, :].unsqueeze(1)

            # Apply transforms
            single_channel = transform(single_channel.squeeze(1)).unsqueeze(1)

            # Extract features
            output = model.forward_features(single_channel)
            feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
            batch_feat.append(feat_temp)

    # Concatenate features from all channels
    embeddings_np = numpy.concatenate(batch_feat, axis=1)
    # embeddings = model.predict(pixels_torch)
    # OpenPhenom: (N, 384)

    return embeddings_np


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply to every pixel in the input regardless of rank/batch shape:
        # the original code's `x[0]` indexed only the first tile in a batch,
        # leaving the rest untouched.
        noise = torch.empty_like(x).uniform_(self.low, self.high)
        return torch.where(x == 255, noise, x)


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


if __name__ == "__main__":
    # address = "ipc:///tmp/vit.ipc"
    address = sys.argv[1]

    logger.add(address.split("/")[-1])

    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
