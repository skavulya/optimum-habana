# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
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
import gc
import unittest

import torch
from trl import DDPOConfig

from optimum.habana import GaudiConfig
from optimum.habana.trl import GaudiDDPOTrainer, GaudiDefaultDDPOStableDiffusionPipeline


def scorer_function(images, prompts, metadata):
    return torch.randn(1) * 3.0, {}


def prompt_function():
    return ("cabbages", {})


class GaudiDDPOTrainerTester(unittest.TestCase):
    """
    Test the GaudiDDPOTrainer class.
    """

    def setUp(self):
        self.ddpo_config = DDPOConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            per_prompt_stat_tracking_buffer_size=32,
            sample_num_batches_per_epoch=2,
            sample_batch_size=2,
            mixed_precision=None,
            save_freq=1000000,
        )
        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"

        gaudi_config = GaudiConfig()
        pipeline = GaudiDefaultDDPOStableDiffusionPipeline(
            pretrained_model,
            pretrained_model_revision=pretrained_revision,
            use_lora=False,
            gaudi_config=gaudi_config,
            use_habana=True,
            use_hpu_graphs=False
        )

        self.trainer = GaudiDDPOTrainer(
            self.ddpo_config,
            scorer_function,
            prompt_function,
            pipeline,
            gaudi_config=gaudi_config,
            use_habana=True,
            use_hpu_graphs=False
        )

        return super().setUp()

    def tearDown(self) -> None:
        gc.collect()

    def test_loss(self):
        advantage = torch.tensor([-1.0])
        clip_range = 0.0001
        ratio = torch.tensor([1.0])
        loss = self.trainer.loss(advantage, clip_range, ratio)
        assert loss.item() == 1.0

    def test_generate_samples(self):
        samples, output_pairs = self.trainer._generate_samples(1, 2)
        assert len(samples) == 1
        assert len(output_pairs) == 1
        assert len(output_pairs[0][0]) == 2

    def test_calculate_loss(self):
        samples, _ = self.trainer._generate_samples(1, 2)
        sample = samples[0]

        latents = sample["latents"][0, 0].unsqueeze(0)
        next_latents = sample["next_latents"][0, 0].unsqueeze(0)
        log_probs = sample["log_probs"][0, 0].unsqueeze(0)
        timesteps = sample["timesteps"][0, 0].unsqueeze(0)
        prompt_embeds = sample["prompt_embeds"]
        advantage = torch.tensor([1.0], device=prompt_embeds.device)

        assert latents.shape == (1, 4, 64, 64)
        assert next_latents.shape == (1, 4, 64, 64)
        assert log_probs.shape == (1,)
        assert timesteps.shape == (1,)
        assert prompt_embeds.shape == (2, 77, 32)
        loss, approx_kl, clipfrac = self.trainer.calculate_loss(
            latents, timesteps, next_latents, log_probs, advantage, prompt_embeds
        )

        assert torch.isfinite(loss.cpu())


class GaudiDDPOTrainerWithLoRATester(GaudiDDPOTrainerTester):
    """
    Test the GaudiDDPOTrainer class.
    """

    def setUp(self):
        self.ddpo_config = DDPOConfig(
            num_epochs=2,
            train_gradient_accumulation_steps=1,
            per_prompt_stat_tracking_buffer_size=32,
            sample_num_batches_per_epoch=2,
            sample_batch_size=2,
            mixed_precision=None,
            save_freq=1000000,
        )
        pretrained_model = "hf-internal-testing/tiny-stable-diffusion-torch"
        pretrained_revision = "main"

        pipeline = GaudiDefaultDDPOStableDiffusionPipeline(
            pretrained_model, pretrained_model_revision=pretrained_revision, use_lora=True
        )

        self.trainer = GaudiDDPOTrainer(self.ddpo_config, scorer_function, prompt_function, pipeline)

        return super().setUp()
