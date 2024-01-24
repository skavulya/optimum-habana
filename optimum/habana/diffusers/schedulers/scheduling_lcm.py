# Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import register_to_config
from diffusers.utils import logging
from diffusers.schedulers import LCMScheduler
from diffusers.schedulers.scheduling_lcm import LCMSchedulerOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class GaudiLCMScheduler(LCMScheduler):
    """
    Extends [Diffusers' LCMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/lcm) to run optimally on Gaudi:
    - All time-dependent parameters are generated at the beginning
    - At each time step, tensors are rolled to update the values of the time-dependent parameters

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        original_inference_steps (`int`, *optional*, defaults to 50):
            The default number of inference steps used to generate a linearly-spaced timestep schedule, from which we
            will ultimately take `num_inference_steps` evenly spaced timesteps to form the final timestep schedule.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False` to make the last step use step 0 for the previous alpha product like in Stable
            Diffusion.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        timestep_scaling (`float`, defaults to 10.0):
            The factor the timesteps will be multiplied by when calculating the consistency model boundary conditions
            `c_skip` and `c_out`. Increasing this will decrease the approximation error (although the approximation
            error at the default of `10.0` is already pretty small).
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        original_inference_steps: int = 50,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        timestep_scaling: float = 10.0,
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            original_inference_steps,
            clip_sample,
            clip_sample_range,
            set_alpha_to_one,
            steps_offset,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            sample_max_value,
            timestep_spacing,
            timestep_scaling,
            rescale_betas_zero_snr
        )
        self._initial_timestep = None
        self.reset_timestep_dependent_params()

    def reset_timestep_dependent_params(self):
        self.are_timestep_dependent_params_set = False
        self.alpha_prod_t_list = []
        self.alpha_prod_t_prev_list = []

    def get_params(self, timestep: Union[int, torch.IntTensor]):
        """
        Initialize the time-dependent parameters, and retrieve the time-dependent
        parameters at each timestep. The tensors are rolled in a separate function
        at the end of the scheduler step in case parameters are retrieved multiple
        times in a timestep, e.g., when scaling model inputs and in the scheduler step.

        Args:
            timestep (`int`):
                The current discrete timestep in the diffusion chain. Optionally used to
                initialize parameters in cases which start in the middle of the
                denoising schedule (e.g. for image-to-image)
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        if not self.are_timestep_dependent_params_set:
            timesteps = self.timesteps[self.step_index:]
            for t in timesteps:
                prev_t = self.previous_timestep(t)
                alpha_prod_t = self.alphas_cumprod[t]
                alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

                self.alpha_prod_t_list.append(alpha_prod_t)
                self.alpha_prod_t_prev_list.append(alpha_prod_t_prev)

            self.alpha_prod_t_list = torch.stack(self.alpha_prod_t_list)
            self.alpha_prod_t_prev_list = torch.stack(self.alpha_prod_t_prev_list)
            self.are_timestep_dependent_params_set = True

        alpha_prod_t = self.alpha_prod_t_list[0]
        alpha_prod_t_prev = self.alpha_prod_t_prev_list[0]

        return alpha_prod_t, alpha_prod_t_prev

    def roll_params(self):
        """
        Roll tensors to update the values of the time-dependent parameters at each timestep.
        """
        if self.are_timestep_dependent_params_set:
            self.alpha_prod_t_list = torch.roll(self.alpha_prod_t_list, shifts=-1, dims=0)
            self.alpha_prod_t_prev_list = torch.roll(self.alpha_prod_t_prev_list, shifts=-1, dims=0)
        else:
            raise ValueError("Time-dependent parameters should be set first.")
        return

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LCMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        # Reset step index at start or when processing next batches
        if (self.step_index is None) or (self.step_index >= self.num_inference_steps):
            self._init_step_index(timestep)

        # 1. Get time-dependent parameters
        alpha_prod_t, alpha_prod_t_prev = self.get_params(timestep)

        # 2. compute alphas, betas
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.config.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )

        # 5. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            predicted_original_sample = self._threshold_sample(predicted_original_sample)
        elif self.config.clip_sample:
            predicted_original_sample = predicted_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 6. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            # torch.randn is broken on HPU so running it on CPU
            device = model_output.device
            rand_device = "cpu" if device.type == "hpu" else device
            noise = torch.randn(
                model_output.shape, generator=generator, device=rand_device, dtype=denoised.dtype
            )
            if device.type == "hpu":
                noise = noise.to(device)

            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        # upon completion increase step index by one
        self._step_index += 1
        self.roll_params()

        if not return_dict:
            return (prev_sample, denoised)

        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)

