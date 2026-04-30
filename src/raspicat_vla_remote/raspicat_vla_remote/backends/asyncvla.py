"""AsyncVLA cloud backend (Plan 2A).

Loads the OpenVLA-OFT backbone + ProprioProjector + L1RegressionActionHead +
Proj_Actiontokens from NHirose/AsyncVLA_release. Runs the full forward pass,
projects the action token hidden states with Proj_Actiontokens, and returns
the resulting `(NUM_ACTIONS_CHUNK=8, 1024)` tensor as the cloud->edge
ActionEmbedding payload. The edge consumes this with its own Edge_adapter
(see :class:`AsyncVLAEdgeAdapter` in raspicat_vla_edge.adapters.asyncvla).

The action_head is loaded but not used at inference (Plan 2A keeps the edge
in the loop). It's instantiated mainly to keep the checkpoint footprint
honest and to support a future "no edge model" fallback that mirrors the
OmniVLA Path 1 mode.

Note: importing prismatic.models.small_head pulls vint_train at module load
time. The MBRA submodule's `train/` directory must be on PYTHONPATH (set by
``Dockerfile.asyncvla``).
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch

from ._checkpoints import load_checkpoint
from .base import ModelInfoDict, VLABackend
from .omnivla_data_transform import build_inference_batch, determine_modality_id


_LOG = logging.getLogger(__name__)


class AsyncVLABackend(VLABackend):
    """Cloud backend running the AsyncVLA forward pass + Proj_Actiontokens."""

    def __init__(
        self,
        *,
        vla_path: str,
        resume_step: int = 750000,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.bfloat16,
        num_images_in_input: int = 2,
    ) -> None:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor, PrismaticProcessor,
        )
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.models.projectors import ProprioProjector
        from prismatic.models.small_head import Proj_Actiontokens
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
        from transformers import (
            AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
        )

        self._vla_path = vla_path
        self._resume_step = resume_step
        self._device = torch.device(device)
        self._dtype = dtype
        self._num_images_in_input = num_images_in_input
        self._prompt_builder_cls = PurePromptBuilder

        self._action_dim = int(ACTION_DIM)
        self._num_actions_chunk = int(NUM_ACTIONS_CHUNK)
        self._pose_dim = int(POSE_DIM)
        # AsyncVLA's Proj_Actiontokens hidden_dim is 1024 (set in run_asyncvla.py:660).
        self._cloud_action_dim = 1024

        AutoConfig.register('openvla', OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1, exist_ok=True)

        _LOG.info('loading AsyncVLA processor + backbone from %s', vla_path)
        self._processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self._vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._vla.vision_backbone.set_num_images_in_input(num_images_in_input)
        self._vla = self._vla.to(dtype=dtype, device=self._device).eval()

        _LOG.info('loading auxiliary heads (step=%d)', resume_step)
        self._pose_projector = ProprioProjector(
            llm_dim=self._vla.llm_dim, proprio_dim=self._pose_dim,
        ).to(self._device).eval()
        self._pose_projector.load_state_dict(
            load_checkpoint('pose_projector', vla_path, resume_step, device=str(self._device)),
        )

        self._action_head = L1RegressionActionHead_idcat(
            input_dim=self._vla.llm_dim,
            hidden_dim=self._vla.llm_dim,
            action_dim=self._action_dim,
        ).to(dtype).to(self._device).eval()
        self._action_head.load_state_dict(
            load_checkpoint('action_head', vla_path, resume_step, device=str(self._device)),
        )

        # AsyncVLA-only: cloud action projector that compresses
        # (B, NUM_ACTIONS_CHUNK*ACTION_DIM, llm_dim) -> (B, NUM_ACTIONS_CHUNK, 1024).
        self._action_proj = Proj_Actiontokens(
            input_dim=self._vla.llm_dim,
            hidden_dim=self._vla.llm_dim,
            action_dim=self._cloud_action_dim,
        ).to(dtype).to(self._device).eval()
        self._action_proj.load_state_dict(
            load_checkpoint('action_proj', vla_path, resume_step, device=str(self._device)),
        )

        self._num_patches = (
            self._vla.vision_backbone.get_num_patches()
            * self._vla.vision_backbone.get_num_images_in_input()
        ) + 1   # +1 for goal_pose token

        self._action_tokenizer = ActionTokenizer(self._processor.tokenizer)
        _LOG.info(
            'AsyncVLA ready (num_patches=%d, NUM_ACTIONS_CHUNK=%d, ACTION_DIM=%d, '
            'cloud_action_dim=%d)',
            self._num_patches, self._num_actions_chunk, self._action_dim,
            self._cloud_action_dim,
        )

    @torch.no_grad()
    def warmup(self, num_iters: int = 1) -> None:
        dummy = PIL.Image.new('RGB', (224, 224))
        for _ in range(max(1, num_iters)):
            self.infer(
                current_image=dummy, past_image=dummy,
                lang_instruction='warmup',
                goal_image=None, goal_pose_xy_theta=(0.0, 0.0, 0.0),
            )

    @torch.no_grad()
    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image] = None,  # AsyncVLA cloud doesn't use past
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        from prismatic.training.train_utils import (
            get_current_action_mask, get_next_actions_mask,
        )

        t0 = time.monotonic()

        modality_id_int = determine_modality_id(
            has_lang=bool(lang_instruction),
            has_pose=goal_pose_xy_theta is not None,
            has_image_goal=goal_image is not None,
        )
        modality_id = torch.as_tensor([modality_id_int], dtype=torch.float32)

        # AsyncVLA's data_transformer_asyncvla is byte-identical to OmniVLA's
        # data_transformer_omnivla -- reuse Plan 2B's helper directly.
        batch = build_inference_batch(
            current_image=current_image,
            goal_image=goal_image,
            lang_instruction=lang_instruction,
            goal_pose_xy_theta=goal_pose_xy_theta,
            action_tokenizer=self._action_tokenizer,
            processor=self._processor,
            prompt_builder_cls=self._prompt_builder_cls,
            pose_dim=self._pose_dim,
            num_actions_chunk=self._num_actions_chunk,
            action_dim=self._action_dim,
        )

        with torch.autocast('cuda', dtype=self._dtype):
            output = self._vla(
                input_ids=batch['input_ids'].to(self._device),
                attention_mask=batch['attention_mask'].to(self._device),
                pixel_values=batch['pixel_values'].to(self._dtype).to(self._device),
                modality_id=modality_id.to(self._dtype).to(self._device),
                labels=batch['labels'].to(self._device),
                output_hidden_states=True,
                proprio=batch['goal_pose'].to(self._dtype).to(self._device),
                proprio_projector=self._pose_projector,
                use_film=False,
            )

        gt_token_ids = batch['labels'][:, 1:].to(self._device)
        cur_mask = get_current_action_mask(gt_token_ids)
        next_mask = get_next_actions_mask(gt_token_ids)

        last_hidden = output.hidden_states[-1]
        text_hidden = last_hidden[:, self._num_patches:-1]
        actions_hidden = (
            text_hidden[cur_mask | next_mask]
            .reshape(1, self._num_actions_chunk * self._action_dim, -1)
            .to(self._dtype)
        )

        # AsyncVLA-specific: project the action hidden states down to the
        # (NUM_ACTIONS_CHUNK, 1024) tensor that the edge's Edge_adapter expects.
        projected = self._action_proj.predict_action(
            actions_hidden.detach(),
            modality_id.to(self._dtype).to(self._device),
        )
        arr = (
            projected
            .detach()
            .reshape(self._num_actions_chunk, self._cloud_action_dim)
            .to(torch.float32)
            .cpu()
            .numpy()
        )

        return arr, {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
            'modality_id': modality_id_int,
        }

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='NHirose/AsyncVLA_release',
            model_version=f'asyncvla-step{self._resume_step}',
            num_tokens=self._num_actions_chunk,
            embed_dim=self._cloud_action_dim,
            device=str(self._device),
            ready=True,
        )
