"""
Production inference layer for the suitability CNN (Phase 2.8.6).

This module is the production counterpart to the training-time code in
``ai.cnn`` / ``ai.dataloader`` / ``ai.training``. It loads a *registered,
trained* suitability model through ``ai.registry.ModelRegistry`` and runs
deterministic, payload-free inference on a single image:

- the input is a PIL image only; no payload bytes, plaintext/ciphertext,
  password, or key is ever received;
- the image is reduced to the Z domain ``Z = image & 0xFE`` (every channel LSB
  cleared) exactly as in ``ai.cnn.z_domain_array`` / ``ai.dataloader``;
- the Z-domain array is normalized to float in [0, 1] and fed to the network
  in the *exact* (N, H, W, 3) layout the model was trained on;
- the network's output map is validated and returned as a ``SuitabilityMap``
  with the input's own HxW and a deterministic ranking.

Determinism
-----------
The same artifact plus the same image always yields the identical
``SuitabilityMap``: model parameters are immutable after load (the model is
reconstructed fresh from the validated artifact on every call), the
preprocessing is a pure function of the image bytes, there is no random
sampling, and the ranking tie-breaks by ``(-score, y, x)``.

The extraction invariant (the Phase 2.8.7 prerequisite)
--------------------------------------------------------
Because prediction is defined purely on ``Z(image)``, and ``Z`` is the LSB-
cleared analysis domain that the Phase 2.6 embedder/extractor and the
Phase 2.8.2 label all share:

    Z(cover) == Z(stego)   =>   predict(cover) == predict(stego)

for the same model. A cover and a stego image that differ only in LSBs map to
the same Z domain and therefore the same suitability map, so the inference layer
is safe to reuse for steganographic suitability even before BWH3 exists.

Design constraints
------------------
- Thin: builds on ``ModelRegistry`` / ``ModelArtifact`` / ``SuitabilityCNN``;
  it does NOT reimplement model reconstruction or the artifact format.
- Does not modify the input image, embed/extract anything, or depend on BWH3.
- Does not use the training/test datasets, nor any network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ai.artifact import ARTIFACT_SCHEMA_VERSION, ModelArtifact
from ai.cnn import SuitabilityCNN, z_domain_array

if TYPE_CHECKING:
    from ai.registry import ModelRegistry
    from app.core.evaluation import FloatArray


class InferenceError(Exception):
    """Raised when inference cannot run or its output fails validation.

    Covers invalid input images, missing/incompatible/tampered models, and
    invalid model output. Registry and artifact validation errors
    (``RegistryError`` / ``ArtifactError``) propagate unchanged — they are
    already specific — so a caller can distinguish "registration problem" from
    "inference problem".
    """


@dataclass(frozen=True)
class SuitabilityMap:
    """Validated HxW embedding-suitability result for one image.

    Attributes:
        scores: ``(H, W)`` float64 array with every value in [0, 1]; index as
            ``scores[y][x]``.
        height: Spatial height H (== input image height).
        width: Spatial width W (== input image width).
    """

    scores: FloatArray
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.scores.ndim != 2 or self.scores.shape != (self.height, self.width):
            msg = (
                f"SuitabilityMap scores shape {getattr(self.scores, 'shape', None)} "
                f"does not match ({self.height}, {self.width})."
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(self.scores)):
            msg = "SuitabilityMap contains non-finite (NaN/Inf) scores."
            raise ValueError(msg)
        if float(self.scores.min()) < 0.0 or float(self.scores.max()) > 1.0:
            msg = (
                f"SuitabilityMap scores out of [0, 1]: "
                f"min={float(self.scores.min())}, max={float(self.scores.max())}."
            )
            raise ValueError(msg)

    def ranking(self) -> list[tuple[int, int]]:
        """Pixel coordinates ``(y, x)`` in descending suitability order.

        Ties are broken deterministically by ascending ``(y, x)``, i.e. the
        full sort key is ``(-score, y, x)``: higher scores first, and among
        equal scores the top-left pixel wins. The result is therefore fully
        determined by ``scores`` alone.

        Returns:
            A list of ``(y, x)`` integer coordinates, ``scores[ranking()[0]]``
            being the most suitable pixel.
        """
        ys, xs = np.meshgrid(
            np.arange(self.height),
            np.arange(self.width),
            indexing="ij",
        )
        y_flat = ys.reshape(-1)
        x_flat = xs.reshape(-1)
        score_flat = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        # lexsort orders by the LAST key first, so -score (descending) is the
        # primary key, then y, then x — exactly the (-score, y, x) tie-break.
        order = np.lexsort((x_flat, y_flat, -score_flat))
        return [(int(y_flat[i]), int(x_flat[i])) for i in order]

    def top_k(self, k: int) -> list[tuple[int, int]]:
        """The ``k`` most suitable pixel coordinates ``(y, x)``.

        Raises:
            ValueError: If ``k`` is negative.
        """
        if k < 0:
            msg = f"top_k requires a non-negative k, got {k}."
            raise ValueError(msg)
        return self.ranking()[:k]


#: Architecture identifier this inference layer is built for. A registered
#: artifact whose architecture does not match is rejected as incompatible.
_SUPPORTED_ARCHITECTURE = "suitability_cnn_v1"


class SuitabilityPredictor:
    """Thin deterministic inference service over a ``ModelRegistry``.

    The predictor holds no model state between calls: each ``predict`` reloads
    and reconstructs the requested model from the registry, so model parameters
    are immutable after load and every call re-validates artifact integrity
    (checksum, schema, finiteness).

    Example:
        registry = ModelRegistry(Path("models"))
        predictor = SuitabilityPredictor(registry)
        result = predictor.predict(image, model_name="suitability_cnn", version="v1")
    """

    def __init__(self, registry: ModelRegistry) -> None:
        """Bind the predictor to a ``ModelRegistry``."""
        self._registry = registry

    @property
    def registry(self) -> ModelRegistry:
        """The registry this predictor loads models from."""
        return self._registry

    def predict(
        self,
        image: Image.Image,
        *,
        model_name: str,
        version: str,
    ) -> SuitabilityMap:
        """Run deterministic inference for `image` with a registered model.

        Args:
            image: A PIL image (any mode; converted to RGB internally). The
                image is never mutated.
            model_name: Registry model name (the stable model identifier).
            version: Registry version identifier.

        Returns:
            A validated ``SuitabilityMap`` with the input's own HxW.

        Raises:
            InferenceError: If `image` is invalid, the model output is invalid,
                or the architecture is incompatible with this inference layer.
            RegistryError: If `model_name`/`version` is missing or invalid
                (never silently falls back to another model).
            ArtifactError: If the artifact is invalid, corrupted, or fails
                checksum/schema/parameter validation on load.
        """
        artifact = self._registry.load(model_name, version)
        self._check_architecture(artifact)
        model = artifact.reconstruct_model()
        return _run_inference(image, model)

    @staticmethod
    def _check_architecture(artifact: ModelArtifact) -> None:
        """Reject artifacts whose architecture this layer cannot run.

        Raises:
            InferenceError: If the artifact's architecture name, depth, or conv
                layer shapes are not exactly the supported baseline.
        """
        arch = artifact.metadata.architecture
        if arch.name != _SUPPORTED_ARCHITECTURE:
            msg = (
                f"Incompatible architecture {arch.name!r}; "
                f"this inference layer supports {_SUPPORTED_ARCHITECTURE!r}."
            )
            raise InferenceError(msg)
        # Cross-check the conv-layer shapes via the CNN's own architecture so the
        # supported layout stays a single source of truth.
        from ai.cnn import _ARCHITECTURE

        # Normalize conv_layers for comparison (JSON deserialization converts tuples to lists)
        artifact_conv_layers = tuple(tuple(layer) for layer in arch.conv_layers)
        if arch.depth != len(_ARCHITECTURE) or artifact_conv_layers != _ARCHITECTURE:
            msg = (
                f"Incompatible architecture layout (depth={arch.depth}, "
                f"conv_layers={arch.conv_layers}); expected depth={len(_ARCHITECTURE)}, "
                f"conv_layers={_ARCHITECTURE}."
            )
            raise InferenceError(msg)
        if artifact.metadata.schema_version != ARTIFACT_SCHEMA_VERSION:
            msg = (
                f"Unsupported artifact schema version "
                f"{artifact.metadata.schema_version}; expected {ARTIFACT_SCHEMA_VERSION}."
            )
            raise InferenceError(msg)


def predict(
    registry: ModelRegistry,
    image: Image.Image,
    *,
    model_name: str,
    version: str,
) -> SuitabilityMap:
    """Module-level convenience: ``predict(registry, image, model_name=..., version=...)``.

    Equivalent to ``SuitabilityPredictor(registry).predict(image, ...)``.
    """
    return SuitabilityPredictor(registry).predict(image, model_name=model_name, version=version)


def _run_inference(image: Image.Image, model: SuitabilityCNN) -> SuitabilityMap:
    """Preprocess `image` and run `model` forward to a validated ``SuitabilityMap``.

    Preprocessing (the inference-domain convention):
    1. convert to RGB (handles grayscale / RGBA / palette modes);
    2. clear every channel LSB — ``Z = image & 0xFE``;
    3. normalize uint8 -> float in [0, 1];

    The network consumes the exact (1, H, W, 3) layout it was trained on.

    Raises:
        InferenceError: If `image` is invalid, or the model output is not a
            finite HxW map in [0, 1] with the input's spatial size.
    """
    if not isinstance(image, Image.Image):
        msg = f"predict requires a PIL image, got {type(image).__name__}."
        raise InferenceError(msg)

    try:
        z = z_domain_array(image)
    except ValueError as exc:
        msg = f"Invalid input image: {exc}"
        raise InferenceError(msg) from exc

    height, width = z.shape[0], z.shape[1]
    if height < 1 or width < 1:
        msg = f"Image must have positive width and height, got ({height}, {width})."
        raise InferenceError(msg)

    # uint8 Z-domain -> float [0, 1], NHWC batch exactly matching training.
    # Model expects float64 (the CNN's internal dtype); keep consistent.
    x = np.asarray(z, dtype=np.float64)[None, ...] / 255.0
    output = model.forward(x)  # (1, H, W)
    scores = np.asarray(output[0], dtype=np.float64)

    if scores.shape != (height, width):
        msg = (
            f"Model output shape {scores.shape} does not match input "
            f"({height}, {width}) (expected HxW)."
        )
        raise InferenceError(msg)
    if not np.all(np.isfinite(scores)):
        msg = "Model produced non-finite (NaN/Inf) output."
        raise InferenceError(msg)
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        # The network ends in sigmoid, so values outside [0, 1] mean a corrupted
        # or incompatible model; reject rather than clip (clipping would mask it).
        msg = f"Model output out of [0, 1]: min={float(scores.min())}, max={float(scores.max())}."
        raise InferenceError(msg)

    return SuitabilityMap(scores=scores, height=height, width=width)
