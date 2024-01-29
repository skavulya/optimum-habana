from trl.import_utils import is_diffusers_available

if is_diffusers_available():
    from .modeling_sd_base import (
        GaudiDefaultDDPOStableDiffusionPipeline,
    )
