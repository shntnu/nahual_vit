"""Model deployment for generic transformers."""

import sys
from functools import partial

import numpy
import pynng
import torch
import transformers
import trio
from loguru import logger
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

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
        OpenPhenom is channel-agnostic — pass however many channels you have
        (registered guardrail is ``(None, 16)``, i.e. no channel padding).
        Tile size 256 is what Recursion used in the paper; we don't enforce
        it but the embedding is most comparable at that resolution.

    Server-side normalization (applied here)
    ----------------------------------------
    None — ``model.predict`` runs the model's own internal preprocessing on
    raw uint8 pixels. Pre-scaling to [0, 1] or z-scoring will degrade
    embeddings.

    Output
    ------
    (N, 384) — a single 384-d embedding per tile.
    """

    _, input_channels, _, *input_yx = pixels.shape

    validate_input_shape(input_yx, expected_yx)

    pixels = pad_channel_dim(pixels, expected_channels)

    pixels_torch = torch.from_numpy(pixels).float().to(device)

    with torch.no_grad():
        embeddings = model.predict(pixels_torch)
        # OpenPhenom: (N, 384)

        embeddings_np = embeddings.cpu().detach().numpy()

    return embeddings_np


if __name__ == "__main__":
    # address = "ipc:///tmp/vit.ipc"
    address = sys.argv[1]

    logger.add(address.split("/")[-1])

    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
