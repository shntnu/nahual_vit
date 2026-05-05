"""
This example uses a server within the environment defined on
`https://github.com/afermg/nahual_vit.git`. nahual_vit serves multiple
models from per-file entrypoints under `src/vit/` (one server file per
model — there is NO single `server.py` at the repo root).

Run with either:
    nix run github:afermg/nahual_vit -- ipc:///tmp/morphem.ipc
or, from a local checkout:
    nix develop --command bash -c "python src/vit/morphem.py ipc:///tmp/morphem.ipc"
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("vit")
address = "ipc:///tmp/morphem.ipc"

# %%Load models server-side
parameters = dict(
    model_name="CaicedoLab/MorphEm",
    # device=0, # optional
)
response = setup(parameters, address=address)

# %% Define custom data
tile_size = 256

# channel can be < 6 and the model will pad
input_shape = (2, 6, 1, tile_size, tile_size)
data = numpy.random.random_sample(input_shape)
result = process(data, address=address)
