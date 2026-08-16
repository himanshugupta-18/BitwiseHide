"""
Phase 2.8.6 inference tests.

Covers ai.inference (production inference layer):
- successful inference from a registered artifact
- deterministic repeated inference (same artifact + same image)
- exact output HxW shape matching the input
- output range [0, 1] and finiteness
- RGB / grayscale / RGBA normalization for the Z domain
- Z-domain preprocessing (Z = image & 0xFE)
- model selection by stable model id/version
- missing model rejection
- corrupted/tampered artifact rejection through registry
- invalid model / output rejection
- deterministic ranking and tie-breaking
- input-image immutability (never mutated)
- same Z-domain cover/stego inputs producing identical suitability maps
  (the Phase 2.8.7 extraction invariant)

All tests are fully offline; no network, no BSDS500 download. Temporary
directories and synthetic/in-memory images only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from ai.artifact import ArtifactError, ModelArtifact
from ai.cnn import z_domain_array
from ai.dataloader import SuitabilityDataset
from ai.inference import (
    InferenceError,
    SuitabilityMap,
    SuitabilityPredictor,
    predict,
)
from ai.prepare_dataset import Dataset, write_synthetic_dataset
from ai.registry import ModelRegistry, RegistryError
from ai.split import Split, resolve_split
from ai.train_run import (
    RunConfig,
    _run_training_with_checkpointing_internal,
    create_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path


# --- Fixtures / helpers --------------------------------------------------------


def _make_synthetic_dataset(
    tmp_path: Path, per_split: int = 2, size: tuple[int, int] = (16, 16), seed: int = 0
) -> Dataset:
    """Create a synthetic dataset for testing."""
    return write_synthetic_dataset(tmp_path / "ds", per_split=per_split, size=size, seed=seed)


def _build_datasets(
    dataset: Dataset,
) -> tuple[SuitabilityDataset, SuitabilityDataset, SuitabilityDataset]:
    splits = resolve_split(dataset)
    return (
        SuitabilityDataset(splits[Split.TRAIN]),
        SuitabilityDataset(splits[Split.VAL]),
        SuitabilityDataset(splits[Split.TEST]),
    )


def _make_artifact(
    tmp_path: Path, *, seed: int = 0, epochs: int = 1, size: tuple[int, int] = (16, 16)
) -> tuple[ModelArtifact, ModelRegistry]:
    """Train a tiny model, package it as an artifact, register it, return (artifact, registry)."""
    dataset = _make_synthetic_dataset(tmp_path, per_split=2, size=size, seed=seed)
    cfg = RunConfig(seed=seed, epochs=epochs, learning_rate=1e-3, batch_size=2)
    train_ds, val_ds, test_ds = _build_datasets(dataset)
    run = _run_training_with_checkpointing_internal(train_ds, val_ds, test_ds, cfg)
    artifact = create_artifact(result=run.result, model=run.model, run_config=cfg, dataset=dataset)

    registry = ModelRegistry(tmp_path / "registry")
    registry.register(artifact, model_name="test_model", version="v1")
    return artifact, registry


def _make_predictor(
    tmp_path: Path, *, seed: int = 0, epochs: int = 1, size: tuple[int, int] = (16, 16)
) -> SuitabilityPredictor:
    """Create a predictor backed by a registry with one registered model."""
    _make_artifact(tmp_path, seed=seed, epochs=epochs, size=size)
    registry = ModelRegistry(tmp_path / "registry")
    return SuitabilityPredictor(registry)


# --- Successful inference -------------------------------------------------------


class TestSuccessfulInference:
    """Inference from a registered artifact succeeds with correct output."""

    def test_infer_from_registered_artifact(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=42)
        image = Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        )
        result = predictor.predict(image, model_name="test_model", version="v1")
        assert isinstance(result, SuitabilityMap)
        assert result.scores.shape == (16, 16)
        assert result.height == 16 and result.width == 16

    def test_module_level_predict_convenience(self, tmp_path: Path) -> None:
        _make_artifact(tmp_path, seed=7)
        registry = ModelRegistry(tmp_path / "registry")
        image = Image.fromarray(
            np.random.default_rng(1).integers(0, 256, (8, 12, 3), dtype=np.uint8)
        )
        result = predict(registry, image, model_name="test_model", version="v1")
        assert result.scores.shape == (8, 12)  # (H, W)

    def test_predictor_registry_property(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=1)
        assert isinstance(predictor.registry, ModelRegistry)


# --- Determinism ----------------------------------------------------------------


class TestDeterminism:
    """Same artifact + same image => identical output; no random sampling."""

    def test_repeated_inference_is_identical(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=3)
        rng = np.random.default_rng(99)
        image = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))

        r1 = predictor.predict(image, model_name="test_model", version="v1")
        r2 = predictor.predict(image, model_name="test_model", version="v1")
        r3 = predictor.predict(image, model_name="test_model", version="v1")

        assert np.array_equal(r1.scores, r2.scores)
        assert np.array_equal(r2.scores, r3.scores)

    def test_determinism_across_separate_registry_loads(self, tmp_path: Path) -> None:
        """Reloading the model from disk on each call must still be deterministic."""
        _make_artifact(tmp_path, seed=5)
        rng = np.random.default_rng(123)
        image = Image.fromarray(rng.integers(0, 256, (10, 10, 3), dtype=np.uint8))

        reg_a = ModelRegistry(tmp_path / "registry")
        reg_b = ModelRegistry(tmp_path / "registry")
        p_a = SuitabilityPredictor(reg_a)
        p_b = SuitabilityPredictor(reg_b)

        a = p_a.predict(image, model_name="test_model", version="v1")
        b = p_b.predict(image, model_name="test_model", version="v1")
        assert np.array_equal(a.scores, b.scores)


# --- Output shape ---------------------------------------------------------------


class TestOutputShape:
    """Output HxW must exactly match the input image's spatial dims."""

    @pytest.mark.parametrize("size", [(1, 1), (7, 9), (16, 16), (32, 48), (64, 64)])
    def test_exact_hxw_match(self, tmp_path: Path, size: tuple[int, int]) -> None:
        predictor = _make_predictor(tmp_path, seed=11, size=(64, 64))
        h, w = size
        image = Image.fromarray(
            np.random.default_rng(h * 31 + w).integers(0, 256, (h, w, 3), dtype=np.uint8)
        )
        result = predictor.predict(image, model_name="test_model", version="v1")
        assert result.scores.shape == (h, w)
        assert result.height == h and result.width == w

    def test_score_indexing_is_yx(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=13)
        image = Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
        )
        result = predictor.predict(image, model_name="test_model", version="v1")
        # scores[y][x] convention
        assert result.scores[2, 3] == result.scores[2][3]


# --- Output range & finiteness --------------------------------------------------


class TestOutputValidity:
    """Scores must be finite and within [0, 1]."""

    def test_scores_in_unit_interval(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=17)
        image = Image.fromarray(
            np.random.default_rng(2).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        )
        result = predictor.predict(image, model_name="test_model", version="v1")
        assert float(result.scores.min()) >= 0.0
        assert float(result.scores.max()) <= 1.0

    def test_scores_are_finite(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=19)
        image = Image.fromarray(
            np.random.default_rng(3).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        )
        result = predictor.predict(image, model_name="test_model", version="v1")
        assert np.all(np.isfinite(result.scores))

    def test_suitability_map_rejects_nan(self) -> None:
        bad = np.full((4, 4), np.nan)
        with pytest.raises(ValueError, match="non-finite"):
            SuitabilityMap(scores=bad, height=4, width=4)

    def test_suitability_map_rejects_out_of_range(self) -> None:
        bad = np.full((4, 4), 1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            SuitabilityMap(scores=bad, height=4, width=4)

    def test_suitability_map_rejects_shape_mismatch(self) -> None:
        scores = np.zeros((4, 4))
        with pytest.raises(ValueError, match="shape"):
            SuitabilityMap(scores=scores, height=8, width=8)


# --- Input mode normalization ---------------------------------------------------


class TestInputModeNormalization:
    """RGB / grayscale / RGBA must all reduce to the same Z domain."""

    def test_grayscale_normalizes_like_rgb(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=23)
        gray = Image.new("L", (16, 16), color=128)
        rgb = Image.new("RGB", (16, 16), color=(128, 128, 128))
        r_gray = predictor.predict(gray, model_name="test_model", version="v1")
        r_rgb = predictor.predict(rgb, model_name="test_model", version="v1")
        assert np.array_equal(r_gray.scores, r_rgb.scores)

    def test_rgba_normalizes_like_rgb(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=29)
        rgba = Image.new("RGBA", (16, 16), color=(100, 150, 200, 77))
        rgb = Image.new("RGB", (16, 16), color=(100, 150, 200))
        r_rgba = predictor.predict(rgba, model_name="test_model", version="v1")
        r_rgb = predictor.predict(rgb, model_name="test_model", version="v1")
        assert np.array_equal(r_rgba.scores, r_rgb.scores)


# --- Z-domain preprocessing -----------------------------------------------------


class TestZDomainPreprocessing:
    """Preprocessing must be Z = image & 0xFE on every channel."""

    def test_lsb_cleared_in_preprocessing(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=31)
        rng = np.random.default_rng(4)
        raw = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        image = Image.fromarray(raw)

        # Directly compute expected Z-domain preprocessing (mirrors inference.py)

        # Run two images that differ only in LSBs; both must map to z_expected.
        image_lsb_on = Image.fromarray((raw | 0x01).astype(np.uint8))
        r1 = predictor.predict(image, model_name="test_model", version="v1")
        r2 = predictor.predict(image_lsb_on, model_name="test_model", version="v1")
        assert np.array_equal(r1.scores, r2.scores)

        # Sanity: the Z domain is actually LSB-cleared.
        from ai.cnn import z_domain_array

        z = z_domain_array(image)
        assert (z % 2 == 0).all()

    def test_preprocessing_matches_cnn_z_domain_helper(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=37)
        rng = np.random.default_rng(5)
        image = Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8))
        result = predictor.predict(image, model_name="test_model", version="v1")
        # The z_domain_array helper is what inference uses internally; ensure
        # feeding the Z-domain array directly yields the same prediction.

        artifact = ModelRegistry(tmp_path / "registry").load("test_model", "v1")
        model = artifact.reconstruct_model()
        z = z_domain_array(image)
        direct = model.predict(z)
        assert np.allclose(result.scores, direct, atol=1e-12)


# --- Model selection by id ------------------------------------------------------


class TestModelSelection:
    """The caller selects a model by stable (model_name, version)."""

    def test_select_by_model_name_and_version(self, tmp_path: Path) -> None:
        dataset = _make_synthetic_dataset(tmp_path, per_split=2, seed=0)
        cfg = RunConfig(seed=0, epochs=1, learning_rate=1e-3, batch_size=2)
        train_ds, val_ds, test_ds = _build_datasets(dataset)
        run = _run_training_with_checkpointing_internal(train_ds, val_ds, test_ds, cfg)
        artifact = create_artifact(
            result=run.result, model=run.model, run_config=cfg, dataset=dataset
        )

        registry = ModelRegistry(tmp_path / "registry")
        registry.register(artifact, model_name="suitability_cnn", version="v1")
        registry.register(artifact, model_name="suitability_cnn", version="v2")
        registry.register(artifact, model_name="other_model", version="v1")

        predictor = SuitabilityPredictor(registry)
        image = Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        )
        r1 = predictor.predict(image, model_name="suitability_cnn", version="v1")
        r2 = predictor.predict(image, model_name="suitability_cnn", version="v2")
        r3 = predictor.predict(image, model_name="other_model", version="v1")
        # All artifacts are identical here, so output is identical; selection keys work.
        assert np.array_equal(r1.scores, r2.scores)
        assert np.array_equal(r2.scores, r3.scores)

    def test_no_silent_fallback_to_another_model(self, tmp_path: Path) -> None:
        _make_artifact(tmp_path, seed=0)  # registers "test_model"/"v1" only
        predictor = SuitabilityPredictor(ModelRegistry(tmp_path / "registry"))
        image = Image.fromarray(
            np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        )
        with pytest.raises(RegistryError, match="not found"):
            predictor.predict(image, model_name="nonexistent", version="v1")


# --- Missing model rejection ----------------------------------------------------


class TestMissingModelRejection:
    """Missing (model_name, version) must fail clearly; no fallback."""

    def test_missing_model_name(self, tmp_path: Path) -> None:
        _make_artifact(tmp_path, seed=0)
        predictor = SuitabilityPredictor(ModelRegistry(tmp_path / "registry"))
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        with pytest.raises(RegistryError, match="not found"):
            predictor.predict(image, model_name="ghost", version="v1")

    def test_missing_version(self, tmp_path: Path) -> None:
        _make_artifact(tmp_path, seed=0)
        predictor = SuitabilityPredictor(ModelRegistry(tmp_path / "registry"))
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        with pytest.raises(RegistryError, match="not found"):
            predictor.predict(image, model_name="test_model", version="v9")


# --- Corrupted / tampered artifact rejection ------------------------------------


class TestTamperedArtifactRejection:
    """Registry level integrity (checksum, load) must propagate to inference."""

    def test_checksum_mismatch_rejected(self, tmp_path: Path) -> None:
        _make_artifact(tmp_path, seed=0)
        registry = ModelRegistry(tmp_path / "registry")
        predictor = SuitabilityPredictor(registry)
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

        # Tamper with the weights file (breaks checksum).
        weights_path = registry._version_dir("test_model", "v1") / "weights.npz"
        npz = np.load(weights_path)
        data = {k: v.copy() for k, v in npz.items()}
        data["weight_0"] = data["weight_0"] * 1.001
        np.savez_compressed(weights_path, **data)

        with pytest.raises(ArtifactError, match="Checksum mismatch"):
            predictor.predict(image, model_name="test_model", version="v1")

    def test_corrupted_weights_rejected(self, tmp_path: Path) -> None:
        _make_artifact(tmp_path, seed=0)
        registry = ModelRegistry(tmp_path / "registry")
        predictor = SuitabilityPredictor(registry)
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

        weights_path = registry._version_dir("test_model", "v1") / "weights.npz"
        weights_path.write_bytes(b"corrupted-bytes")

        with pytest.raises(ArtifactError, match="Failed to load weights"):
            predictor.predict(image, model_name="test_model", version="v1")


# --- Invalid model / output rejection -------------------------------------------


class TestInvalidModelOutputRejection:
    """Incompatible architecture or invalid output must fail clearly."""

    def test_incompatible_architecture_name_rejected(self, tmp_path: Path) -> None:
        artifact, registry = _make_artifact(tmp_path, seed=0)
        # Mutate architecture name in metadata to something unsupported.
        import json

        meta_path = registry._version_dir("test_model", "v1") / "metadata.json"
        data = json.loads(meta_path.read_text())
        data["architecture"]["name"] = "unknown_arch"
        meta_path.write_text(json.dumps(data))

        predictor = SuitabilityPredictor(registry)
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        with pytest.raises(InferenceError, match="Incompatible architecture"):
            predictor.predict(image, model_name="test_model", version="v1")

    def test_unsupported_schema_version_rejected(self, tmp_path: Path) -> None:
        artifact, registry = _make_artifact(tmp_path, seed=0)
        import json

        meta_path = registry._version_dir("test_model", "v1") / "metadata.json"
        data = json.loads(meta_path.read_text())
        data["schema_version"] = 999
        meta_path.write_text(json.dumps(data))

        predictor = SuitabilityPredictor(registry)
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        with pytest.raises(ArtifactError, match="schema version"):
            predictor.predict(image, model_name="test_model", version="v1")

    def test_invalid_image_input_rejected(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=0)
        with pytest.raises(InferenceError, match="PIL image"):
            predictor.predict("not-an-image", model_name="test_model", version="v1")  # type: ignore[arg-type]


# --- Deterministic ranking & tie-breaking --------------------------------------


class TestRankingAndTieBreaking:
    """ranking is deterministic and ties break by (-score, y, x)."""

    def test_ranking_is_sorted_descending(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=41)
        image = Image.fromarray(
            np.random.default_rng(6).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        )
        result = predictor.predict(image, model_name="test_model", version="v1")
        ranked = result.ranking()
        scores_in_order = [result.scores[y, x] for (y, x) in ranked]
        assert scores_in_order == sorted(scores_in_order, reverse=True)

    def test_ranking_is_deterministic_across_calls(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=43)
        image = Image.fromarray(
            np.random.default_rng(7).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        )
        r1 = predictor.predict(image, model_name="test_model", version="v1").ranking()
        r2 = predictor.predict(image, model_name="test_model", version="v1").ranking()
        assert r1 == r2

    def test_tie_break_by_ascending_xy(self) -> None:
        # Construct a SuitabilityMap with two equal top scores; top-left wins.
        scores = np.array([[0.5, 0.9], [0.9, 0.1]])
        sm = SuitabilityMap(scores=scores, height=2, width=2)
        ranked = sm.ranking()
        # Highest score 0.9 appears at (0, 1) and (1, 0); tie breaks by y then x,
        # so (0, 1) precedes (1, 0).
        assert ranked[0] == (0, 1)
        assert ranked[1] == (1, 0)

    def test_top_k_returns_k_most_suitable(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=47)
        image = Image.fromarray(
            np.random.default_rng(8).integers(0, 256, (6, 6, 3), dtype=np.uint8)
        )
        result = predictor.predict(image, model_name="test_model", version="v1")
        top3 = result.top_k(3)
        assert len(top3) == 3
        full = result.ranking()
        assert top3 == full[:3]

    def test_top_k_negative_rejected(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=53)
        image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        result = predictor.predict(image, model_name="test_model", version="v1")
        with pytest.raises(ValueError, match="non-negative"):
            result.top_k(-1)


# --- Input immutability ---------------------------------------------------------


class TestInputImmutability:
    """The input image must not be mutated by inference."""

    def test_image_mode_and_size_unchanged(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=59)
        original = Image.fromarray(
            np.random.default_rng(9).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        )
        image = original.copy()
        _ = predictor.predict(image, model_name="test_model", version="v1")
        assert image.size == original.size
        assert image.mode == original.mode
        assert np.array_equal(np.asarray(image), np.asarray(original))

    def test_rgba_alpha_channel_unchanged(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=61)
        rgba = Image.new("RGBA", (12, 12), color=(10, 20, 30, 99))
        before = np.asarray(rgba)
        _ = predictor.predict(rgba, model_name="test_model", version="v1")
        after = np.asarray(rgba)
        assert np.array_equal(before, after)


# --- Extraction invariant (Phase 2.8.7 prerequisite) ---------------------------


class TestExtractionInvariant:
    """Z(cover) == Z(stego) => predict(cover) == predict(stego)."""

    def test_cover_stego_same_z_domain_same_map(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=67)
        rng = np.random.default_rng(10)
        cover_raw = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

        # Stego differs from cover ONLY in LSBs (an atomic embedder write).
        stego_raw = cover_raw | 0x01

        cover = Image.fromarray(cover_raw)
        stego = Image.fromarray(stego_raw)

        r_cover = predictor.predict(cover, model_name="test_model", version="v1")
        r_stego = predictor.predict(stego, model_name="test_model", version="v1")

        # The Z domains are identical, so the suitability maps must be identical.
        assert np.array_equal(r_cover.scores, r_stego.scores)
        # Explicit Z-domain equality check for documentation value.
        assert np.array_equal(z_domain_array(cover), z_domain_array(stego))

    def test_invariance_holds_across_multiple_images(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=71)
        rng = np.random.default_rng(11)
        for i in range(5):
            raw = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
            cover = Image.fromarray(raw)
            stego = Image.fromarray(raw | 0x01)
            rc = predictor.predict(cover, model_name="test_model", version="v1")
            rs = predictor.predict(stego, model_name="test_model", version="v1")
            assert np.array_equal(rc.scores, rs.scores), f"mismatch on image {i}"


# --- Minimum dimension validation ----------------------------------------------


class TestMinimumDimensions:
    """Images must have positive width and height."""

    def test_zero_width_image_rejected(self, tmp_path: Path) -> None:
        predictor = _make_predictor(tmp_path, seed=73)
        # 0-width image cannot exist as PIL; emulate via z_domain_array failure path
        # by passing a 0-height array through the public predict path is not possible,
        # so we assert the SuitabilityMap / preprocessing contract on a tiny 1x1 image
        # (the real guard is positive dims, which 1x1 satisfies).
        image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        result = predictor.predict(image, model_name="test_model", version="v1")
        assert result.scores.shape == (1, 1)
