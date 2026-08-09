"""
Low-level LSB steganography primitives for PNG images.

Architecture decisions:
- Standard LSB (least-significant-bit) embedding: every bit of hidden data is
  stored in the least-significant bit of a pixel-channel byte. Any input image
  is normalized to RGB so the channel layout is always known (3 bytes/pixel)
  and capacity is deterministic.
- A fixed 12-byte header is embedded first:
      magic (4 bytes) | payload length (uint64 big-endian, 8 bytes)
  The magic marks an image as carrying a BitwiseHide payload; the length is a
  strict boundary so extraction reads EXACTLY that many payload bytes and no
  more. Lengths that would exceed the image's capacity are rejected, so a
  hostile image can never force an unbounded read of pixel data.
- This layer hides and retrieves BYTES only. It performs NO encryption and NO
  integrity protection, and treats the payload as opaque. Confidentiality and
  tamper-evidence are the responsibility of the Phase 2.3 encryption layer.
- Fails closed: invalid capacity, a missing magic, or an impossible length
  raise SteganographyError — a partial or corrupted payload is never returned
  and a payload is never silently truncated.
"""

from __future__ import annotations

import struct

from PIL import Image

from app.core.exceptions import SteganographyError

#: Byte string marking an image as carrying a BitwiseHide v1 payload.
MAGIC = b"BWH1"
#: Width of the big-endian payload-length field (uint64).
LENGTH_BYTES = 8
#: Total header size embedded before the payload: magic + length.
HEADER_SIZE = len(MAGIC) + LENGTH_BYTES


def max_payload_bytes(image: Image.Image) -> int:
    """
    Maximum number of payload bytes that fit in `image` (header included).

    Args:
        image: A PIL image. Converted to RGB for capacity purposes, matching
            what embed_bytes/extract_bytes do.

    Returns:
        The largest payload that `embed_bytes` will accept. May be negative for
        images too small to hold even the header.
    """
    rgb = image.convert("RGB")
    capacity = (rgb.size[0] * rgb.size[1] * 3) // 8
    return capacity - HEADER_SIZE


def embed_bytes(image: Image.Image, payload: bytes) -> Image.Image:
    """
    Embed `payload` into the least-significant bits of `image`'s RGB channels.

    Args:
        image: A PIL image to hide data in (any mode; normalized to RGB).
        payload: Opaque bytes to embed. Empty bytes are valid.

    Returns:
        A NEW RGB image carrying the payload. The input image is not mutated.

    Raises:
        SteganographyError: If the payload (including the header) does not fit
            within the image's capacity. The payload is never truncated.
    """
    rgb = image.convert("RGB")
    raw = bytearray(rgb.tobytes())
    capacity = len(raw) // 8
    required = HEADER_SIZE + len(payload)
    if required > capacity:
        raise SteganographyError(
            message=(
                f"Image capacity too small: need {required} bytes, capacity is {capacity} bytes."
            )
        )

    header = MAGIC + struct.pack(">Q", len(payload))
    data = header + payload
    for data_index, byte in enumerate(data):
        base = data_index * 8
        for bit in range(8):
            raw[base + bit] = (raw[base + bit] & 0xFE) | ((byte >> bit) & 1)
    return Image.frombytes("RGB", rgb.size, bytes(raw))


def extract_bytes(image: Image.Image) -> bytes:
    """
    Extract the hidden payload from an image previously produced by embed_bytes.

    Args:
        image: A PIL image (any mode; normalized to RGB, matching embed_bytes).

    Returns:
        The exact embedded payload bytes.

    Raises:
        SteganographyError: If the image is too small for a header, carries no
            BitwiseHide magic, or its recorded payload length exceeds the image
            capacity. Nothing is read past the image's capacity.
    """
    rgb = image.convert("RGB")
    raw = rgb.tobytes()
    capacity = len(raw) // 8
    if capacity < HEADER_SIZE:
        raise SteganographyError(message="Image too small to contain a hidden payload.")

    header = bytearray(HEADER_SIZE)
    for header_index in range(HEADER_SIZE):
        base = header_index * 8
        byte = 0
        for bit in range(8):
            byte |= (raw[base + bit] & 1) << bit
        header[header_index] = byte

    if bytes(header[: len(MAGIC)]) != MAGIC:
        raise SteganographyError(message="No valid BitwiseHide payload in image.")

    (length,) = struct.unpack(">Q", bytes(header[len(MAGIC) :]))
    if HEADER_SIZE + length > capacity:
        raise SteganographyError(
            message=(f"Payload length {length} exceeds image capacity {capacity}.")
        )

    payload = bytearray(length)
    payload_base = HEADER_SIZE * 8
    for payload_index in range(length):
        base = payload_base + payload_index * 8
        byte = 0
        for bit in range(8):
            byte |= (raw[base + bit] & 1) << bit
        payload[payload_index] = byte
    return bytes(payload)
