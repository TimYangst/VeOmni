# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Collator-side vision helpers shared by the Qwen VL family patch configs.

``qwen3_5``, ``qwen3_5_moe``, ``qwen3_vl`` and ``qwen3_vl_moe`` all expose the
same two CPU-side collate hooks, and the bodies are model-agnostic: they only
touch batch keys (``pixel_values``, ``*_grid_thw``) that every member of the
family shares. Defining them once here keeps the four patch configs from
drifting apart.

Each config registers them with ``config.add_helper(...)``, which emits the
source verbatim into that model's generated modeling. Keep them free of
model-specific symbols: helpers bypass patchgen's per-patch ``name_map``, so a
reference to a class such as ``Qwen3VLModel`` would land unrenamed in every
other family's generated file.
"""

import torch


def collate_multimodal_metadata(batch, sp_pad):
    """Derive ``multimodal_metadata`` for the Qwen3-VL-family ViT.

    Module-level so ``get_metadata_collate_func`` can hand it to VeOmni's
    collator as a picklable callable (mirrors ``get_position_id``). Runs
    purely on CPU inside the collator after SP padding — every value it
    produces (CPU int tensors / Python ints / lists) is consumed by the ViT
    forward without a host-device sync.

    ``batch`` is the packed (+ SP-padded) batch dict; ``sp_pad`` maps
    ``pixel_values`` / ``pixel_values_videos`` to the number of patch rows
    the SP collator appended. Mutates ``batch`` in place, writing
    ``batch["multimodal_metadata"]``.
    """
    md = {}
    # ViT varlen-attention metadata, derived from the HF processor's
    # ``*_grid_thw`` CPU LongTensor (packed across the batch by the collator
    # via DataCollateInfo pack_dim=0). ``.tolist()`` here is a pure-CPU op —
    # the collator runs in dataloader workers, no host-device sync.
    # Temporal unroll: each (t, h, w) expands to ``t`` cu steps of ``h * w``.
    if "pixel_values_merged" in batch:
        # The pre-slice hook (`merge_pixel_streams`) replaced the raw pixel
        # streams with one merged stream (image rows first, then video rows):
        # emit ONE cu_seqlens over image frames then video frames, with a
        # single sp-pad tail (zero when SP is off), plus the global image
        # patch-row count Model.forward needs to split the feature stream.
        merged_grid_list = []
        cu = [0]
        max_hw = 0
        n_image_rows = 0
        for grid_key in ("image_grid_thw", "video_grid_thw"):
            grid = batch.get(grid_key)
            if grid is None:
                continue
            grid_list = grid.tolist() if torch.is_tensor(grid) else grid
            if not grid_list:
                continue
            merged_grid_list.extend(grid_list)
            for t, h, w in grid_list:
                hw = h * w
                max_hw = max(max_hw, hw)
                for _ in range(t):
                    cu.append(cu[-1] + hw)
                if grid_key == "image_grid_thw":
                    n_image_rows += t * hw
        pad = sp_pad.get("pixel_values_merged", 0)
        if pad > 0:
            cu.append(cu[-1] + pad)
            max_hw = max(max_hw, pad)
        if merged_grid_list:
            md["merged_grid_thw_list"] = merged_grid_list
            md["vit_merged_cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device="cpu")
            md["vit_merged_max_seqlen"] = max_hw
            md["vit_merged_n_image_rows"] = n_image_rows
    else:
        for modality, grid_key, pad_key in (
            ("image", "image_grid_thw", "pixel_values"),
            ("video", "video_grid_thw", "pixel_values_videos"),
        ):
            grid = batch.get(grid_key)
            if grid is None:
                continue
            grid_list = grid.tolist() if torch.is_tensor(grid) else grid
            if not grid_list:
                continue
            md[f"{modality}_grid_thw_list"] = grid_list
            cu = [0]
            max_hw = 0
            for t, h, w in grid_list:
                hw = h * w
                max_hw = max(max_hw, hw)
                for _ in range(t):
                    cu.append(cu[-1] + hw)
            # SP-pad tail: the collator zero-pads pixel_values to SP-divisible;
            # those patches become one synthetic "image" so varlen attention
            # treats them as an independent sequence (mirrors the position_ids==0
            # text-side SP-pad convention). Discarded after the per-rank slice.
            pad = sp_pad.get(pad_key, 0)
            if pad > 0:
                cu.append(cu[-1] + pad)
                max_hw = max(max_hw, pad)
            # device='cpu': this runs in CPU dataloader workers — pin to CPU so a
            # global torch.set_default_device('cuda') can't misallocate it.
            md[f"vit_{modality}_cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device="cpu")
            md[f"vit_{modality}_max_seqlen"] = max_hw

    if md:
        batch["multimodal_metadata"] = md


def merge_pixel_streams(batch):
    """VeOmni pre-slice collate hook (``PreSliceCollateFunc`` in
    ``veomni/data/data_collator.py``): merge the image and video pixel streams
    into one ``pixel_values_merged`` stream so the vision tower runs exactly
    once per rank per step.

    Runs BEFORE the collator's per-key SP pad/slice, which is what makes it
    correct under SP: the Ulysses sequence exchange then sees one
    globally-ordered stream (image rows first, then video rows — the order
    ``collate_multimodal_metadata`` and Model.forward assume), whereas
    concatenating two independently-sliced rank-local streams afterwards would
    misorder it. Also runs with SP disabled, where there is no slice but the
    single merged layout is what Model.forward expects.

    Runs on CPU inside DataLoader workers; module-level for picklability.
    Mutates ``batch`` in place; the ``*_grid_thw`` tensors stay untouched."""
    pixel_values = batch.pop("pixel_values", None)
    pixel_values_videos = batch.pop("pixel_values_videos", None)
    streams = [stream for stream in (pixel_values, pixel_values_videos) if stream is not None]
    if not streams:
        return
    batch["pixel_values_merged"] = torch.cat(streams, dim=0) if len(streams) > 1 else streams[0]
