"""
BWH3 (BitwiseHide 3) AI-guided LSB embedding for PNG images (Phase 2.8.7).

This layer decides WHERE to hide payload bits using an AI suitability model
(ai.inference.SuitabilityPredictor) instead of the Phase 2.5 complexity analysis.
The actual bit-level embedding mechanics REUSE the existing BWH1/Phase 2.4
primitives (core.steganography) — only the candidate selection changes.

Architecture decisions:
- REUSES ai.inference.SuitabilityPredictor for the suitability map.
  The map is computed ONCE from the original cover image (Z = cover & 0xFE)
  and remains FIXED for the entire embedding operation.
- The actual bit manipulation is implemented directly in this module
  (not delegated to core.steganography) to allow AI-guided candidate positions.
- A dedicated header marks BWH3 payloads and carries the minimum metadata
  extraction needs to reproduce the candidate ordering:
      magic (4) | version (1) | flags (1) | model_name_len (2, BE)
      | model_name (variable) | model_version_len (2, BE)
      | model_version (variable) | payload length (4, BE, uint32)
  The magic "BWH3" intentionally differs from Phase 2.4's "BWH1" and
  Phase 2.6's "BWH2" so the three formats never cross-parse.
- The model_name and model_version in the header allow extraction to
  reproduce the EXACT candidate ordering by loading the same model.
- This layer hides and retrieves BYTES only. It performs NO encryption and
  NO integrity protection, and treats the payload as opaque. Confidentiality
  and tamper-evidence are the responsibility of the Phase 2.3 encryption layer.
- Fails closed: a missing/corrupted magic, an unsupported version/flags,
  invalid model metadata, an impossible payload length, or insufficient
  capacity raise BWH3EmbeddingError. Payloads are never truncated and
  partial payloads are never returned.

Determinism:
The same cover image + same model artifact/version + same config ALWAYS
yields the same stego image. The candidate ordering is derived from the
SuitabilityMap ranking (descending score, ties broken by y, then x, then
channel R/G/B). The ordering is computed once from the original cover.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from ai.inference import SuitabilityMap, SuitabilityPredictor


#: Byte string marking an image as carrying a BitwiseHide BWH3 payload.
#: Deliberately distinct from core.steganography.MAGIC ("BWH1") and
#: adaptive_embedding.ADAPTIVE_MAGIC ("BWH2") so the formats never cross-parse.
BWH3_MAGIC = b"BWH3"

#: Version of the BWH3 embedding format understood by this module.
BWH3_FORMAT_VERSION = 1

#: Reserved header flags; version 1 requires this to be zero.
BWH3_FORMAT_FLAGS = 0

#: Maximum length for model_name and model_version strings in the header.
_MAX_MODEL_STRING_LEN = 255

#: On-disk header layout (dynamic size due to variable-length strings):
#:   magic (4) | version (1) | flags (1) | model_name_len (2, BE)
#:   | model_name (model_name_len) | model_version_len (2, BE)
#:   | model_version (model_version_len) | payload length (4, BE, uint32)
# Fixed portion: 4 + 1 + 1 + 2 + 2 + 4 = 14 bytes
# Plus model_name_len + model_version_len

#: Header size without the variable-length strings.
_BWH3_FIXED_HEADER_SIZE = 14

#: Maximum total header size (fixed + max strings).
_MAX_BWH3_HEADER_SIZE = _BWH3_FIXED_HEADER_SIZE + 2 * _MAX_MODEL_STRING_LEN

#: Total header size in bits (the header occupies the leading LSB positions).
#: This is the MAXIMUM possible header size in bits; actual header size
#: depends on string lengths.
_MAX_BWH3_HEADER_BITS = _MAX_BWH3_HEADER_SIZE * 8


@dataclass(frozen=True)
class BWH3EmbedConfig:
    """Immutable configuration for BWH3 AI-guided embedding.

    Attributes:
        model_name: Registry model name (the stable model identifier).
        model_version: Registry version identifier.
        predictor: The SuitabilityPredictor to use for generating the map.
    """

    model_name: str
    model_version: str
    predictor: SuitabilityPredictor

    def __post_init__(self) -> None:
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if len(self.model_name) > _MAX_MODEL_STRING_LEN:
            raise ValueError(f"model_name exceeds maximum length of {_MAX_MODEL_STRING_LEN}")
        if not self.model_version or not isinstance(self.model_version, str):
            raise ValueError("model_version must be a non-empty string")
        if len(self.model_version) > _MAX_MODEL_STRING_LEN:
            raise ValueError(f"model_version exceeds maximum length of {_MAX_MODEL_STRING_LEN}")
        if self.predictor is None:
            raise ValueError("predictor must not be None")


class BWH3EmbeddingError(Exception):
    """Raised when BWH3 embedding or extraction fails.

    Covers invalid input images, missing/incompatible models, capacity
    exceeded, invalid headers, and extraction failures.
    """

    def __init__(self, message: str = "BWH3 embedding error.") -> None:
        self.message = message
        super().__init__(self.message)


def _encode_bwh3_header(
    *,
    model_name: str,
    model_version: str,
    payload_length: int,
) -> bytes:
    """Serialize the BWH3 header."""
    name_bytes = model_name.encode("utf-8")
    version_bytes = model_version.encode("utf-8")
    if len(name_bytes) > _MAX_MODEL_STRING_LEN:
        raise BWH3EmbeddingError(
            f"model_name too long: {len(name_bytes)} > {_MAX_MODEL_STRING_LEN}"
        )
    if len(version_bytes) > _MAX_MODEL_STRING_LEN:
        raise BWH3EmbeddingError(
            f"model_version too long: {len(version_bytes)} > {_MAX_MODEL_STRING_LEN}"
        )

    # Fixed portion: magic(4) version(1) flags(1) name_len(2) version_len(2) payload_len(4)
    fixed = struct.pack(
        ">4sBBHHI",
        BWH3_MAGIC,
        BWH3_FORMAT_VERSION,
        BWH3_FORMAT_FLAGS,
        len(name_bytes),
        len(version_bytes),
        payload_length,
    )
    return fixed + name_bytes + version_bytes


def _decode_bwh3_header(header: bytes) -> tuple[str, str, int]:
    """
    Validate and parse a raw BWH3 header, returning (model_name, model_version, length).

    Raises:
        BWH3EmbeddingError: If the magic, version, flags, or string lengths are invalid.
            The length is not bounds-checked here — that happens against the image's
            capacity in extract_bytes.
    """
    if len(header) < _BWH3_FIXED_HEADER_SIZE:
        raise BWH3EmbeddingError(message="BWH3 header truncated: missing fixed portion")

    magic, version, flags, name_len, version_len, payload_length = struct.unpack(
        ">4sBBHHI", header[:_BWH3_FIXED_HEADER_SIZE]
    )

    if magic != BWH3_MAGIC:
        raise BWH3EmbeddingError(message="No valid BWH3 payload in image.")
    if version != BWH3_FORMAT_VERSION:
        raise BWH3EmbeddingError(message=f"Unsupported BWH3 payload version: {version}.")
    if flags != 0:
        raise BWH3EmbeddingError(message=f"Unsupported BWH3 payload flags: {flags}.")
    if name_len > _MAX_MODEL_STRING_LEN:
        raise BWH3EmbeddingError(message=f"BWH3 model_name length out of range: {name_len}.")
    if version_len > _MAX_MODEL_STRING_LEN:
        raise BWH3EmbeddingError(message=f"BWH3 model_version length out of range: {version_len}.")

    expected_total = _BWH3_FIXED_HEADER_SIZE + name_len + version_len
    if len(header) != expected_total:
        raise BWH3EmbeddingError(
            message=f"BWH3 header size mismatch: expected {expected_total}, got {len(header)}"
        )

    name_bytes = header[_BWH3_FIXED_HEADER_SIZE : _BWH3_FIXED_HEADER_SIZE + name_len]
    version_bytes = header[
        _BWH3_FIXED_HEADER_SIZE + name_len : _BWH3_FIXED_HEADER_SIZE + name_len + version_len
    ]

    try:
        model_name = name_bytes.decode("utf-8")
        model_version = version_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BWH3EmbeddingError(message="BWH3 header string not valid UTF-8") from exc

    return model_name, model_version, payload_length


def _generate_candidate_positions(
    suitability_map: SuitabilityMap, width: int, height: int, header_bits: int
) -> list[int]:
    """
    Generate flat LSB positions for payload bits in AI-guided order.

    The order follows the SuitabilityMap ranking (most suitable pixel first);
    the three RGB channels of each pixel are used in fixed R,G,B order, and
    the header's reserved leading positions are skipped.

    This is the single source of truth for both embed and extract, so the
    order is always reproducible from the suitability map alone.

    Args:
        suitability_map: The validated suitability map from the AI model.
        width: Image width in pixels.
        height: Image height in pixels.
        header_bits: Number of bits occupied by the header (so payload positions
            start after this offset).

    Returns:
        A list of flat byte indices (0 to width*height*3 - 1) representing
        LSB positions for payload bits, in embedding order.
    """
    positions: list[int] = []
    for row, col in suitability_map.ranking():
        if row >= height or col >= width:
            # Safety: ranking should never produce out-of-bounds coordinates
            continue
        pixel = (row * width + col) * 3
        for channel in range(3):  # R, G, B in fixed order
            position = pixel + channel
            if position >= header_bits:
                positions.append(position)
    return positions


def max_payload_bytes(image: Image.Image, config: BWH3EmbedConfig) -> int:
    """
    Maximum payload bytes that fit in `image` with the given BWH3 configuration.

    Uses the actual header size based on the model name and version strings
    to provide an accurate capacity calculation.

    Args:
        image: A PIL image. Converted to RGB for capacity purposes.
        config: BWH3 embedding configuration containing model info.

    Returns:
        The largest payload that embed_bytes will accept. May be negative for
        images too small to hold even the header.
    """
    rgb = image.convert("RGB")
    total_bits = rgb.size[0] * rgb.size[1] * 3

    # Build the header to know its exact size for this specific config
    header = _encode_bwh3_header(
        model_name=config.model_name,
        model_version=config.model_version,
        payload_length=0,  # zero payload to get header size
    )
    header_bits = len(header) * 8

    return (total_bits - header_bits) // 8


def embed_bytes(
    image: Image.Image,
    payload: bytes,
    *,
    config: BWH3EmbedConfig,
) -> Image.Image:
    """
    Embed `payload` into the LSBs of the most suitable pixels of `image`.

    The embedding order is fully deterministic and self-describing: the header
    records the model identification and payload length, and the order itself
    is the AI suitability ranking of the image with its LSBs cleared.

    Args:
        image: A PIL image to hide data in (any mode; normalized to RGB).
        payload: Opaque bytes to embed. Empty bytes are valid.
        config: BWH3EmbedConfig containing the model and predictor.

    Returns:
        A NEW RGB image carrying the payload. The input image is not mutated.

    Raises:
        BWH3EmbeddingError: If the payload (including the header) does not
            fit within the image's capacity. The payload is never truncated.
        ValueError: If the config is invalid (caller misconfiguration).
    """
    if not isinstance(image, Image.Image):
        raise BWH3EmbeddingError(message="BWH3 embedding requires a PIL image.")
    if not isinstance(payload, bytes):
        raise TypeError(f"Payload must be bytes, got {type(payload).__name__}.")

    # Generate suitability map from the ORIGINAL cover image (not mutated)
    suitability_map = config.predictor.predict(
        image, model_name=config.model_name, version=config.model_version
    )

    rgb = image.convert("RGB")
    width, height = rgb.size
    total_bits = width * height * 3

    # Build the header first to know its exact size
    header = _encode_bwh3_header(
        model_name=config.model_name,
        model_version=config.model_version,
        payload_length=len(payload),
    )
    header_bits = len(header) * 8

    required_bits = header_bits + len(payload) * 8
    if required_bits > total_bits:
        raise BWH3EmbeddingError(
            message=(
                f"Image capacity too small: need {required_bits} bits "
                f"({header_bits} header + {len(payload) * 8} payload), "
                f"capacity is {total_bits} bits."
            )
        )

    # Generate candidate positions using the FIXED suitability map
    positions = _generate_candidate_positions(suitability_map, width, height, header_bits)

    # Ensure we have enough candidate positions for the payload
    payload_bits = len(payload) * 8
    if payload_bits > len(positions):
        raise BWH3EmbeddingError(
            message=(
                f"Insufficient candidate positions: need {payload_bits} bits for payload, "
                f"only {len(positions)} available after header."
            )
        )

    # Work on a copy; only the selected LSBs are ever touched.
    raw = bytearray(rgb.tobytes())

    # Write header at fixed leading positions (0 to header_bits-1)
    for data_index, byte in enumerate(header):
        base = data_index * 8
        for bit in range(8):
            raw[base + bit] = (raw[base + bit] & 0xFE) | ((byte >> bit) & 1)

    # Write payload bits at AI-selected candidate positions
    # We only need the first `payload_bits` positions from the candidate list
    payload_positions = positions[:payload_bits]
    for data_index, byte in enumerate(payload):
        base = data_index * 8
        for bit in range(8):
            position = payload_positions[base + bit]
            raw[position] = (raw[position] & 0xFE) | ((byte >> bit) & 1)

    return Image.frombytes("RGB", rgb.size, bytes(raw))


def extract_bytes(
    image: Image.Image,
    predictor: SuitabilityPredictor,
) -> bytes:
    """
    Extract the BWH3 payload from an image produced by embed_bytes.

    The embedding order is reproduced from the stego image alone: the header
    (read from the fixed leading positions) supplies the model identification
    and payload length. The suitability map is recomputed from the SAME model
    (loaded via the predictor using the header's model_name/version) on the
    image with its LSBs cleared — which is bit-identical to the embedder's
    analysis input, so the candidate positions match exactly.

    Args:
        image: A PIL image (any mode; normalized to RGB, matching embed_bytes).
        predictor: The SuitabilityPredictor to use for recomputing the map.

    Returns:
        The exact embedded payload bytes.

    Raises:
        BWH3EmbeddingError: If the image is too small for a header, carries
            no valid BWH3 magic, has an unsupported version/flags or invalid
            model metadata, or its recorded payload length exceeds the image
            capacity. Nothing is read past the available data and no partial
            payload is returned.
    """
    if not isinstance(image, Image.Image):
        raise BWH3EmbeddingError(message="BWH3 extraction requires a PIL image.")
    if predictor is None:
        raise BWH3EmbeddingError(message="BWH3 extraction requires a SuitabilityPredictor.")

    rgb = image.convert("RGB")
    raw = rgb.tobytes()
    total_bits = len(raw)

    if total_bits < _BWH3_FIXED_HEADER_SIZE * 8:
        raise BWH3EmbeddingError(message="Image too small to contain a BWH3 payload.")

    # Read the fixed portion of the header first (14 bytes) to learn the
    # variable-length string sizes, then read the full header.
    fixed_header = bytearray(_BWH3_FIXED_HEADER_SIZE)
    for i in range(_BWH3_FIXED_HEADER_SIZE):
        base = i * 8
        byte = 0
        for bit in range(8):
            byte |= (raw[base + bit] & 1) << bit
        fixed_header[i] = byte

    # Peek at name_len and version_len from the fixed portion to know the
    # exact total header size.
    _, _, _, name_len, version_len, _ = struct.unpack(">4sBBHHI", bytes(fixed_header))
    if name_len > _MAX_MODEL_STRING_LEN or version_len > _MAX_MODEL_STRING_LEN:
        raise BWH3EmbeddingError(message="BWH3 model metadata length out of range.")

    exact_header_size = _BWH3_FIXED_HEADER_SIZE + name_len + version_len
    if exact_header_size * 8 > total_bits:
        raise BWH3EmbeddingError(message="Image too small to contain a BWH3 payload.")

    # Read the full exact header
    header_bytes = bytearray(exact_header_size)
    for i in range(exact_header_size):
        base = i * 8
        byte = 0
        for bit in range(8):
            byte |= (raw[base + bit] & 1) << bit
        header_bytes[i] = byte

    # Parse the header to get actual header size and payload length
    model_name, model_version, payload_length = _decode_bwh3_header(bytes(header_bytes))

    actual_header_bits = exact_header_size * 8
    if actual_header_bits + payload_length * 8 > total_bits:
        raise BWH3EmbeddingError(message=f"Payload length {payload_length} exceeds image capacity.")

    # Recompute suitability map using the SAME model from the header
    suitability_map = predictor.predict(image, model_name=model_name, version=model_version)

    # Generate candidate positions (same as embed)
    width, height = rgb.size
    positions = _generate_candidate_positions(suitability_map, width, height, actual_header_bits)

    payload_bits = payload_length * 8
    if payload_bits > len(positions):
        raise BWH3EmbeddingError(
            message=(
                f"Insufficient candidate positions: need {payload_bits} bits for payload, "
                f"only {len(positions)} available after header."
            )
        )

    # Read payload bits from AI-selected candidate positions
    payload_positions = positions[:payload_bits]
    out = bytearray(payload_length)
    for data_index in range(payload_length):
        base = data_index * 8
        byte = 0
        for bit in range(8):
            position = payload_positions[base + bit]
            byte |= (raw[position] & 1) << bit
        out[data_index] = byte

    return bytes(out)
