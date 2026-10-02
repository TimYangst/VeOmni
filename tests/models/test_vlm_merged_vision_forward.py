"""Merged image+video vision forward equivalence (exactly-once ViT execution).

The Qwen VL family serves a mixed image+video micro-batch with ONE
vision-tower call: the collator merges the two pixel streams into
``pixel_values_merged`` and Model.forward splits the feature stream back at
``vit_merged_n_image_rows // spatial_merge_unit`` (see the Patch.7 markers in
the patch configs and ``.agents/knowledge/multimodal_metadata.md``). This test
locks the underlying invariant on toy vision towers, CPU-only:

1. the merged call's features match the two per-modality calls (up to GEMM
   batching rounding), including the per-layer deepstack streams;
2. the collator hook pair (``merge_pixel_streams`` then
   ``collate_multimodal_metadata``) builds the expected merged ``cu_seqlens``
   and reproduces the no-metadata ViT result exactly.
"""

import os
from dataclasses import dataclass, field

import pytest
import torch
from transformers import AutoConfig


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Case:
    case_id: str
    toy_config_dir: str
    generated_module: str
    vision_cls_name: str
    vision_overrides: dict = field(default_factory=dict)


_SMALL_VIT = {"depth": 4, "hidden_size": 128, "num_heads": 4, "intermediate_size": 256, "out_hidden_size": 128}

CASES = [
    Case(
        "qwen3_5",
        os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3_5_toy"),
        "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu",
        "Qwen3_5VisionModel",
        dict(_SMALL_VIT),
    ),
    Case(
        "qwen3_5_moe",
        os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3_5_moe_toy"),
        "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu",
        "Qwen3_5MoeVisionModel",
        dict(_SMALL_VIT),
    ),
    Case(
        "qwen3_vl",
        os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3vl_toy"),
        "veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu",
        "Qwen3VLVisionModel",
        # Toy deepstack indexes sit above the toy depth; remap so the
        # deepstack path actually runs.
        {**_SMALL_VIT, "deepstack_visual_indexes": [0, 2]},
    ),
    Case(
        "qwen3_vl_moe",
        os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3vlmoe_toy"),
        "veomni.models.transformers.qwen3_vl_moe.generated.patched_modeling_qwen3_vl_moe_gpu",
        "Qwen3VLMoeVisionModel",
        {**_SMALL_VIT, "deepstack_visual_indexes": [0, 2]},
    ),
]


def _build(case: Case):
    module = pytest.importorskip(case.generated_module)
    if not os.path.isdir(case.toy_config_dir):
        pytest.skip(f"Path not found: {case.toy_config_dir}")
    vision_config = AutoConfig.from_pretrained(case.toy_config_dir).vision_config
    for key, value in case.vision_overrides.items():
        setattr(vision_config, key, value)
    vision_config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = getattr(module, case.vision_cls_name)(vision_config).float().eval()
    return module, model, vision_config


def _make_pixels(vision_config):
    feat_dim = (
        vision_config.in_channels
        * vision_config.temporal_patch_size
        * vision_config.patch_size
        * vision_config.patch_size
    )
    gen = torch.Generator().manual_seed(1)
    image_pixels = torch.randn(1 * 2 * 2, feat_dim, generator=gen)
    image_grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    video_pixels = torch.randn(2 * 2 * 2, feat_dim, generator=gen)
    video_grid = torch.tensor([[2, 2, 2]], dtype=torch.long)
    return image_pixels, image_grid, video_pixels, video_grid


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_merged_vision_forward_matches_split(case):
    _, model, vision_config = _build(case)
    image_pixels, image_grid, video_pixels, video_grid = _make_pixels(vision_config)

    with torch.no_grad():
        image_out = model(image_pixels, grid_thw=image_grid)
        video_out = model(video_pixels, grid_thw=video_grid)
        merged_out = model(
            torch.cat([image_pixels, video_pixels], dim=0),
            grid_thw=torch.cat([image_grid, video_grid], dim=0),
        )

    n_image_features = image_pixels.shape[0] // model.spatial_merge_unit
    torch.testing.assert_close(
        merged_out.pooler_output[:n_image_features], image_out.pooler_output, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        merged_out.pooler_output[n_image_features:], video_out.pooler_output, rtol=1e-5, atol=1e-5
    )

    merged_deepstack = getattr(merged_out, "deepstack_features", None)
    if merged_deepstack:
        assert len(merged_deepstack) == len(image_out.deepstack_features) > 0
        for merged_embed, image_embed, video_embed in zip(
            merged_deepstack, image_out.deepstack_features, video_out.deepstack_features
        ):
            torch.testing.assert_close(merged_embed[:n_image_features], image_embed, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(merged_embed[n_image_features:], video_embed, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_collator_merged_metadata_matches_no_metadata_path(case):
    module, model, vision_config = _build(case)
    image_pixels, image_grid, video_pixels, video_grid = _make_pixels(vision_config)

    # Drive the real collator hooks in pipeline order: the pre-slice merge
    # replaces the raw streams, then the metadata hook keys off the merged one.
    batch = {
        "pixel_values": image_pixels,
        "image_grid_thw": image_grid,
        "pixel_values_videos": video_pixels,
        "video_grid_thw": video_grid,
    }
    module.merge_pixel_streams(batch)
    assert "pixel_values" not in batch and "pixel_values_videos" not in batch
    merged_pixels = batch["pixel_values_merged"]
    assert torch.equal(merged_pixels, torch.cat([image_pixels, video_pixels], dim=0))

    module.collate_multimodal_metadata(batch, {})
    merged_md = batch["multimodal_metadata"]
    assert merged_md["merged_grid_thw_list"] == [[1, 2, 2], [2, 2, 2]]
    # image: one 2x2 frame; video: two 2x2 frames, offset by the image rows.
    # sp_pad is empty here (SP off), so there is no pad-tail segment.
    assert merged_md["vit_merged_cu_seqlens"].tolist() == [0, 4, 8, 12]
    assert merged_md["vit_merged_max_seqlen"] == 4
    assert merged_md["vit_merged_n_image_rows"] == image_pixels.shape[0]
    # Merged mode must not also emit the per-modality keys.
    assert "vit_image_cu_seqlens" not in merged_md and "vit_video_cu_seqlens" not in merged_md

    # Model.forward builds exactly this kwargs dict from the merged metadata.
    merged_vit_kwargs = {
        "vit_metadata": {
            "grid_thw_list": merged_md["merged_grid_thw_list"],
            "cu_seqlens": merged_md["vit_merged_cu_seqlens"],
            "max_seqlen": merged_md["vit_merged_max_seqlen"],
        }
    }
    merged_grid = torch.cat([image_grid, video_grid], dim=0)
    with torch.no_grad():
        out_no_metadata = model(merged_pixels, grid_thw=merged_grid)
        out_metadata = model(merged_pixels, grid_thw=merged_grid, **merged_vit_kwargs)

    assert torch.equal(out_metadata.pooler_output, out_no_metadata.pooler_output)
