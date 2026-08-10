"""
Deterministic adaptive LSB embedding for PNG images (Phase 2.6).

This layer decides WHERE to hide payload bits using Phase 2.5's complexity
analysis (app.core.adaptive.analyze) instead of writing in row-major order:
bits are stored in the RGB-channel LSBs of the most visually complex pixels
first, and smooth regions are used last.

Architecture decisions:

- REUSES app.core.adaptive.analyze verbatim. No Sobel/texture logic is
  duplicated here — the ComplexityMap and its ranked pixel order ARE the
  embedding priority.
- EXTRACTION NEEDS NO ORIGINAL IMAGE. Both embedder and extractor derive the
  embedding order from the SAME deterministic input: the image with every RGB
  channel LSB cleared (Z(image)). Embedding only ever flips LSBs, so
  Z(stego) == Z(cover) exactly — the complexity map, and therefore the bit
  order, computed from either image is bit-identical. The decoder recomputes
  the map from the stego image alone.
- A dedicated 14-byte header marks adaptive payloads and carries the minimum
  metadata extraction needs:
      magic (4) | version (1) | flags (1) | edge_weight (2, BE)
      | texture_weight (2, BE) | payload length (4, BE, uint32)
  The magic "BWH2" intentionally differs from Phase 2.4's sequential "BWH1"
  magic (core.steganography) so the two formats never cross-parse; Phase 2.4's
  header and module are left untouched.
- Analysis weights are stored fixed-point (x256) so extraction re-derives the
  EXACT weights the embedder used — the map, and hence the bit order, matches
  even for non-default weights. Weights are quantized once, before the map is
  computed, so embed and extract always see identical values.
- This layer hides and retrieves BYTES only. It performs NO encryption and NO
  integrity protection and treats the payload as opaque; confidentiality and
  tamper-evidence belong to the Phase 2.3 encryption layer.
- Fails closed: a missing/corrupted magic, an unsupported version/flags,
  invalid analysis metadata, an impossible payload length, or insufficient
  capacity raise AdaptiveEmbeddingError. Payloads are never truncated and
  partial payloads are never returned.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, cast

from PIL import Image

from app.core.adaptive import analyze
from app.core.exceptions import AdaptiveEmbeddingError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.core.adaptive import ComplexityMap

#: Byte string marking an image as carrying a BitwiseHide ADAPTIVE payload.
#: Deliberately distinct from core.steganography.MAGIC ("BWH1") so sequential
#: and adaptive payloads never cross-parse.
ADAPTIVE_MAGIC = b"BWH2"

#: Version of the adaptive embedding format understood by this module.
FORMAT_VERSION = 1

#: Reserved header flags; version 1 requires this to be zero.
FORMAT_FLAGS = 0

#: Fixed-point scale used to store analysis weights (weight * _WEIGHT_SCALE).
_WEIGHT_SCALE = 256

#: Upper bound on analysis weights that the header can represent. 64.0 is far
#: above any sensible relative weight; it keeps the quantized value inside the
#: uint16 field with a large safety margin.
_MAX_WEIGHT = 64.0

#: On-disk header layout:
#:   magic (4) | version (1) | flags (1) | edge_weight (2, big-endian)
#:   | texture_weight (2, big-endian) | payload length (4, big-endian, uint32)
_HEADER_STRUCT = struct.Struct(">4sBBHHI")

#: Total header size in bytes.
HEADER_SIZE = _HEADER_STRUCT.size
#: Total header size in bits (the header occupies the leading LSB positions).
HEADER_BITS = HEADER_SIZE * 8


def _quantize_weight(weight: float) -> int:
    """Validate a caller-supplied weight and quantize it to header form."""
    if weight < 0:
        msg = "Analysis weights must be non-negative."
        raise ValueError(msg)
    if weight > _MAX_WEIGHT:
        msg = f"Analysis weights must not exceed {_MAX_WEIGHT}."
        raise ValueError(msg)
    return round(weight * _WEIGHT_SCALE)


def _prepare_weights(edge_weight: float, texture_weight: float) -> tuple[int, int]:
    """
    Validate and quantize caller-supplied weights, returning their uint16 pair.

    The header can only represent fixed-point (x256) values, so this is the
    authoritative form. The exact floats a decoder recovers from these integers
    are what both embed and extract must feed to analyze() — guaranteeing the
    maps (and therefore the bit order) match even for non-default weights.
    """
    edge_q = _quantize_weight(edge_weight)
    texture_q = _quantize_weight(texture_weight)
    if edge_q == 0 and texture_q == 0:
        msg = "Analysis weights must not both be zero."
        raise ValueError(msg)
    return edge_q, texture_q


def _dequantize_weight(quantized: int) -> float:
    """Recover the float weight a decoder will use from its uint16 form."""
    return quantized / _WEIGHT_SCALE


def _decode_weights(edge_q: int, texture_q: int) -> tuple[float, float]:
    """
    Validate header-stored weights and return the floats a decoder must use.

    Raises:
        AdaptiveEmbeddingError: If both weights are zero or either exceeds the
            representable maximum (a malformed/hostile header).
    """
    if edge_q == 0 and texture_q == 0:
        raise AdaptiveEmbeddingError(message="Header analysis weights are both zero.")
    if edge_q > _MAX_WEIGHT * _WEIGHT_SCALE or texture_q > _MAX_WEIGHT * _WEIGHT_SCALE:
        raise AdaptiveEmbeddingError(message="Header analysis weights are out of range.")
    return _dequantize_weight(edge_q), _dequantize_weight(texture_q)


def _encode_header(*, edge_q: int, texture_q: int, payload_length: int) -> bytes:
    """Serialize the 14-byte adaptive header."""
    return _HEADER_STRUCT.pack(
        ADAPTIVE_MAGIC,
        FORMAT_VERSION,
        FORMAT_FLAGS,
        edge_q,
        texture_q,
        payload_length,
    )


def _decode_header(header: bytes) -> tuple[tuple[float, float], int]:
    """
    Validate and parse a raw header, returning ((edge, texture), length).

    Raises:
        AdaptiveEmbeddingError: If the magic, version, flags, or weights are
            invalid. The length is not bounds-checked here — that happens
            against the image's capacity in extract_bytes.
    """
    magic, version, flags, edge_q, texture_q, length = cast(
        "tuple[bytes, int, int, int, int, int]", _HEADER_STRUCT.unpack(header)
    )
    if magic != ADAPTIVE_MAGIC:
        raise AdaptiveEmbeddingError(message="No valid adaptive payload in image.")
    if version != FORMAT_VERSION:
        raise AdaptiveEmbeddingError(message=f"Unsupported adaptive payload version: {version}.")
    if flags != 0:
        raise AdaptiveEmbeddingError(message=f"Unsupported adaptive payload flags: {flags}.")
    return _decode_weights(edge_q, texture_q), length


def _analysis_input(image: Image.Image) -> Image.Image:
    """
    Return the RGB image with every channel LSB cleared — the shared analysis
    domain (Z(image) in the module docstring).

    Both embed and extract analyze this representation, so the complexity map
    is identical even though the embedded bits differ.
    """
    rgb = image.convert("RGB")
    raw = bytearray(rgb.tobytes())
    for i in range(len(raw)):
        raw[i] &= 0xFE
    return Image.frombytes("RGB", rgb.size, bytes(raw))


def _write_bits(raw: bytearray, data: bytes, start: int) -> None:
    """Write `data`'s bits into the LSB stream of `raw`, starting at `start`."""
    for data_index, byte in enumerate(data):
        base = start + data_index * 8
        for bit in range(8):
            raw[base + bit] = (raw[base + bit] & 0xFE) | ((byte >> bit) & 1)


def _read_bits(raw: bytes, start: int, length_bytes: int) -> bytes:
    """Read `length_bytes` bytes from the LSB stream of `raw`, starting at `start`."""
    out = bytearray(length_bytes)
    for data_index in range(length_bytes):
        base = start + data_index * 8
        byte = 0
        for bit in range(8):
            byte |= (raw[base + bit] & 1) << bit
        out[data_index] = byte
    return bytes(out)


def _write_bits_at_positions(raw: bytearray, positions: Sequence[int], data: bytes) -> None:
    """Write `data`'s bits into the LSBs at the given stream `positions`."""
    for data_index, byte in enumerate(data):
        base = data_index * 8
        for bit in range(8):
            position = positions[base + bit]
            raw[position] = (raw[position] & 0xFE) | ((byte >> bit) & 1)


def _read_bits_at_positions(raw: bytes, positions: Sequence[int], length_bytes: int) -> bytes:
    """Read `length_bytes` bytes from the LSBs at the given stream `positions`."""
    out = bytearray(length_bytes)
    for data_index in range(length_bytes):
        base = data_index * 8
        byte = 0
        for bit in range(8):
            byte |= (raw[positions[base + bit]] & 1) << bit
        out[data_index] = byte
    return bytes(out)


def _payload_bit_positions(map_: ComplexityMap, width: int) -> list[int]:
    """
    Flat LSB positions for payload bits, in adaptive embedding order.

    The order follows the ComplexityMap ranking (most complex pixel first);
    the three RGB channels of each pixel are used in fixed R,G,B order, and the
    header's reserved leading positions are skipped. This is the single source
    of truth for both embed and extract, so the order is always reproducible
    from the map alone — which itself is reproducible from the stego image.
    """
    positions: list[int] = []
    for row, col, _score in map_.ranked:
        pixel = (row * width + col) * 3
        for channel in range(3):
            position = pixel + channel
            if position >= HEADER_BITS:
                positions.append(position)
    return positions


def max_payload_bytes(image: Image.Image) -> int:
    """
    Maximum payload bytes that fit in `image` (adaptive header included).

    Args:
        image: A PIL image. Converted to RGB for capacity purposes, matching
            what embed_bytes/extract_bytes do.

    Returns:
        The largest payload that embed_bytes will accept. May be negative for
        images too small to hold even the header.
    """
    rgb = image.convert("RGB")
    total_bits = rgb.size[0] * rgb.size[1] * 3
    return (total_bits - HEADER_BITS) // 8


def embed_bytes(
    image: Image.Image,
    payload: bytes,
    *,
    edge_weight: float = 1.0,
    texture_weight: float = 1.0,
) -> Image.Image:
    """
    Embed `payload` into the LSBs of the most complex pixels of `image`.

    The embedding order is fully deterministic and self-describing: the header
    (fixed leading LSBs) records the analysis weights and payload length, and
    the order itself is the Phase 2.5 complexity ranking of the image with its
    LSBs cleared.

    Args:
        image: A PIL image to hide data in (any mode; normalized to RGB).
        payload: Opaque bytes to embed. Empty bytes are valid.
        edge_weight: Sobel-edge weight for the Phase 2.5 analysis.
        texture_weight: Local-texture weight for the Phase 2.5 analysis. At
            least one must be positive; both are quantized and stored in the
            header so extraction reproduces the exact embedding order.

    Returns:
        A NEW RGB image carrying the payload. The input image is not mutated.

    Raises:
        AdaptiveEmbeddingError: If the payload (including the header) does not
            fit within the image's capacity. The payload is never truncated.
        ValueError: If the analysis weights are invalid (caller misconfiguration).
    """
    if not isinstance(image, Image.Image):
        raise AdaptiveEmbeddingError(message="Adaptive embedding requires a PIL image.")
    if not isinstance(payload, bytes):
        msg = f"Payload must be bytes, got {type(payload).__name__}."
        raise TypeError(msg)
    rgb = image.convert("RGB")
    width, height = rgb.size
    total_bits = width * height * 3

    required_bits = HEADER_BITS + len(payload) * 8
    if required_bits > total_bits:
        raise AdaptiveEmbeddingError(
            message=(
                f"Image capacity too small: need {required_bits} bits, "
                f"capacity is {total_bits - HEADER_BITS} bits."
            )
        )

    # Quantize the weights once; embed and extract must analyze with identical
    # values, so the map — and therefore the bit order — matches exactly.
    edge_q, texture_q = _prepare_weights(edge_weight, texture_weight)
    cmap = analyze(
        _analysis_input(rgb),
        edge_weight=_dequantize_weight(edge_q),
        texture_weight=_dequantize_weight(texture_q),
    )

    # Work on a copy; only the selected LSBs are ever touched.
    raw = bytearray(rgb.tobytes())
    header = _encode_header(
        edge_q=edge_q,
        texture_q=texture_q,
        payload_length=len(payload),
    )
    _write_bits(raw, header, 0)

    positions = _payload_bit_positions(cmap, width)
    _write_bits_at_positions(raw, positions, payload)

    return Image.frombytes("RGB", rgb.size, bytes(raw))


def extract_bytes(image: Image.Image) -> bytes:
    """
    Extract the adaptive payload from an image produced by embed_bytes.

    The embedding order is reproduced from the stego image alone: the header
    (read from the fixed leading positions) supplies the analysis weights and
    payload length, and the complexity map is recomputed from the image with
    its LSBs cleared — bit-identical to the embedder's analysis input, so the
    bit positions match exactly. No original cover image is required.

    Args:
        image: A PIL image (any mode; normalized to RGB, matching embed_bytes).

    Returns:
        The exact embedded payload bytes (opaque — caller validates them).

    Raises:
        AdaptiveEmbeddingError: If the image is too small for a header, carries
            no valid adaptive magic, has an unsupported version/flags or
            invalid analysis metadata, or its recorded payload length exceeds
            the image capacity. Nothing is read past the available data and no
            partial payload is returned.
    """
    if not isinstance(image, Image.Image):
        raise AdaptiveEmbeddingError(message="Adaptive extraction requires a PIL image.")
    rgb = image.convert("RGB")
    raw = rgb.tobytes()
    total_bits = len(raw)
    if total_bits < HEADER_BITS:
        raise AdaptiveEmbeddingError(message="Image too small to contain an adaptive payload.")

    header = _read_bits(raw, 0, HEADER_SIZE)
    (edge_weight, texture_weight), length = _decode_header(header)

    if HEADER_BITS + length * 8 > total_bits:
        raise AdaptiveEmbeddingError(message=f"Payload length {length} exceeds image capacity.")

    cmap = analyze(
        _analysis_input(rgb),
        edge_weight=edge_weight,
        texture_weight=texture_weight,
    )
    positions = _payload_bit_positions(cmap, rgb.size[0])
    return _read_bits_at_positions(raw, positions, length)
