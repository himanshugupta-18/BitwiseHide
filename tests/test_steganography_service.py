"""
Steganography service + LSB primitive unit tests.

Covers Phase 2.4 behavior:
- exact byte round trips (small, large, binary, empty) through PNG images
- capacity validation — never silently truncates a payload
- fail-closed rejection of malformed/truncated images and non-PNG input
- fail-closed rejection of images without a valid hidden payload (bad magic)
- header integrity: a corrupted magic or an oversized recorded length is rejected
- payload-region corruption is NOT masked here — it is handed to Phase 2.3,
  whose GCM authentication detects it (fail-closed at the right layer)
- an end-to-end Phase 2.3 encrypt -> embed -> extract -> decrypt round trip,
  including wrong-password rejection

No external services are involved; all images are generated in memory.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.core.exceptions import EncryptionError, SteganographyError
from app.core.steganography import (
    HEADER_SIZE,
    LENGTH_BYTES,
    MAGIC,
    embed_bytes,
    extract_bytes,
    max_payload_bytes,
)
from app.schemas.encryption import EncryptedPayload
from app.services.encryption_service import EncryptionService
from app.services.steganography_service import SteganographyService

PASSWORD = "correct-horse-battery-staple"  # noqa: S105 — test fixture, not a real secret
WRONG_PASSWORD = "definitely-not-the-password"  # noqa: S105

# All 256 possible byte values — proves binary fidelity, not just text.
_ALL_BYTES = bytes(range(256))


def _make_png(size: tuple[int, int] = (64, 64)) -> bytes:
    """Create a solid-color RGB PNG in memory."""
    image = Image.new("RGB", size, color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_with_bit_flipped(png: bytes, byte_index: int) -> bytes:
    """Return a copy of the PNG with the LSB of raw RGB byte `byte_index` flipped."""
    image = Image.open(io.BytesIO(png)).convert("RGB")
    raw = bytearray(image.tobytes())
    raw[byte_index] ^= 0x01
    out = Image.frombytes("RGB", image.size, bytes(raw))
    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def png_image() -> bytes:
    return _make_png()


@pytest.fixture
def stego_service() -> SteganographyService:
    return SteganographyService()


@pytest.fixture
def encryption_service() -> EncryptionService:
    return EncryptionService()


class TestRoundTrip:
    """Exact byte round trips through embed/extract."""

    def test_small_payload(self, stego_service: SteganographyService, png_image: bytes) -> None:
        payload = b"hello stego"
        assert (
            stego_service.extract(stego_service.embed(image=png_image, payload=payload)) == payload
        )

    def test_all_byte_values(self, stego_service: SteganographyService, png_image: bytes) -> None:
        stego = stego_service.embed(image=png_image, payload=_ALL_BYTES)
        assert stego_service.extract(stego) == _ALL_BYTES

    def test_larger_payload(self, stego_service: SteganographyService) -> None:
        image = _make_png((128, 128))  # capacity 6144 bytes
        payload = _ALL_BYTES * 8  # 2048 bytes
        stego = stego_service.embed(image=image, payload=payload)
        assert stego_service.extract(stego) == payload

    def test_empty_payload(self, stego_service: SteganographyService, png_image: bytes) -> None:
        stego = stego_service.embed(image=png_image, payload=b"")
        assert stego_service.extract(stego) == b""

    def test_payload_is_not_modified(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        payload = b"untouched-by-stego"
        stego = stego_service.embed(image=png_image, payload=payload)
        assert stego_service.extract(stego) == payload

    def test_repeated_embed_extract(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        for payload in (b"a", b"bb", _ALL_BYTES, b""):
            assert (
                stego_service.extract(stego_service.embed(image=png_image, payload=payload))
                == payload
            )

    def test_embed_is_deterministic(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        """Same image + payload must produce identical stego bytes."""
        first = stego_service.embed(image=png_image, payload=b"data")
        second = stego_service.embed(image=png_image, payload=b"data")
        assert first == second

    def test_embedding_actually_changes_the_image(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        stego = stego_service.embed(image=png_image, payload=b"x" * 100)
        assert stego != png_image


class TestCapacity:
    """Capacity is validated; payloads are never silently truncated."""

    def test_max_payload_bytes_reports_capacity(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        expected = (64 * 64 * 3) // 8 - HEADER_SIZE
        assert max_payload_bytes(image) == expected

    def test_exact_max_payload_fits(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        image = Image.open(io.BytesIO(png_image))
        payload = b"m" * max_payload_bytes(image)
        stego = stego_service.embed(image=png_image, payload=payload)
        assert stego_service.extract(stego) == payload

    def test_one_byte_over_capacity_rejected(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        image = Image.open(io.BytesIO(png_image))
        payload = b"m" * (max_payload_bytes(image) + 1)
        with pytest.raises(SteganographyError, match="capacity"):
            stego_service.embed(image=png_image, payload=payload)

    def test_tiny_image_rejected(self, stego_service: SteganographyService) -> None:
        with pytest.raises(SteganographyError):
            stego_service.embed(image=_make_png((1, 1)), payload=b"anything")


class TestMalformedInput:
    """Malformed or non-PNG input fails closed with SteganographyError."""

    def test_empty_image_rejected(self, stego_service: SteganographyService) -> None:
        with pytest.raises(SteganographyError, match="empty"):
            stego_service.embed(image=b"", payload=b"data")

    def test_non_image_bytes_rejected(self, stego_service: SteganographyService) -> None:
        with pytest.raises(SteganographyError, match="Invalid image"):
            stego_service.embed(image=b"this is not an image at all", payload=b"data")

    def test_non_png_rejected(self, stego_service: SteganographyService) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="JPEG")
        jpeg = buffer.getvalue()
        with pytest.raises(SteganographyError, match="Only PNG"):
            stego_service.embed(image=jpeg, payload=b"data")
        with pytest.raises(SteganographyError, match="Only PNG"):
            stego_service.extract(jpeg)

    def test_truncated_png_rejected(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        stego = stego_service.embed(image=png_image, payload=b"hidden data")
        with pytest.raises(SteganographyError, match="Invalid image"):
            stego_service.extract(stego[: len(stego) // 2])


class TestHeaderIntegrity:
    """Extraction must reject images without a valid, sane header."""

    def test_plain_image_has_no_payload(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        with pytest.raises(SteganographyError, match="No valid BitwiseHide payload"):
            stego_service.extract(png_image)

    def test_corrupted_magic_rejected(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        stego = stego_service.embed(image=png_image, payload=b"secret")
        corrupted = _png_with_bit_flipped(stego, byte_index=0)  # first magic bit
        with pytest.raises(SteganographyError, match="No valid BitwiseHide payload"):
            stego_service.extract(corrupted)

    def test_oversized_length_rejected(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        stego = stego_service.embed(image=png_image, payload=b"small")
        # The length field starts at data byte len(MAGIC); its most significant
        # byte lives in pixel bytes [len(MAGIC)*8, ...]. Flipping its LSB claims
        # a ~2**56-byte payload that can never fit -> rejected by bounds.
        corrupted = _png_with_bit_flipped(stego, byte_index=len(MAGIC) * 8)
        with pytest.raises(SteganographyError, match="exceeds image capacity"):
            stego_service.extract(corrupted)

    def test_tiny_image_has_no_payload(self, stego_service: SteganographyService) -> None:
        # 1x1 RGB = 3 bytes < 12-byte header.
        with pytest.raises(SteganographyError, match="too small"):
            stego_service.extract(_make_png((1, 1)))


class TestPrimitives:
    """Low-level core.steganography behavior."""

    def test_embed_extract_round_trip(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        payload = _ALL_BYTES
        stego = embed_bytes(image, payload)
        assert extract_bytes(stego) == payload

    def test_embed_does_not_mutate_input_image(self, png_image: bytes) -> None:
        image = Image.open(io.BytesIO(png_image))
        before = image.tobytes()
        embed_bytes(image, b"hello")
        assert image.tobytes() == before  # returns a new image, input untouched

    def test_header_layout(self) -> None:
        assert len(MAGIC) == 4
        assert LENGTH_BYTES == 8
        assert len(MAGIC) + LENGTH_BYTES == HEADER_SIZE


class TestOutputImage:
    """The produced PNG must remain a usable, unchanged-size image."""

    def test_output_is_readable_png(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        stego = stego_service.embed(image=png_image, payload=b"hidden data")
        image = Image.open(io.BytesIO(stego))
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (64, 64)


class TestPhase23Integration:
    """End-to-end: EncryptionService -> SteganographyService -> EncryptionService."""

    def test_encrypted_payload_full_round_trip(
        self,
        stego_service: SteganographyService,
        encryption_service: EncryptionService,
        png_image: bytes,
    ) -> None:
        plaintext = "phase-2.4-integration-secret"
        payload = encryption_service.encrypt(plaintext=plaintext, password=PASSWORD)
        stego = stego_service.embed(image=png_image, payload=payload.to_bytes())

        extracted = stego_service.extract(stego)
        restored = EncryptedPayload.from_bytes(extracted)
        assert encryption_service.decrypt(restored, PASSWORD) == plaintext

    def test_extracted_payload_wrong_password_fails(
        self,
        stego_service: SteganographyService,
        encryption_service: EncryptionService,
        png_image: bytes,
    ) -> None:
        payload = encryption_service.encrypt(plaintext="secret", password=PASSWORD)
        stego = stego_service.embed(image=png_image, payload=payload.to_bytes())

        extracted = stego_service.extract(stego)
        restored = EncryptedPayload.from_bytes(extracted)
        with pytest.raises(EncryptionError):
            encryption_service.decrypt(restored, WRONG_PASSWORD)

    def test_corrupted_encrypted_payload_fails_closed(
        self,
        stego_service: SteganographyService,
        encryption_service: EncryptionService,
        png_image: bytes,
    ) -> None:
        """A bit flip in the encrypted payload region must never yield plaintext."""
        plaintext = "corruption-sensitive"
        payload = encryption_service.encrypt(plaintext=plaintext, password=PASSWORD)
        original = payload.to_bytes()
        stego = stego_service.embed(image=png_image, payload=original)

        # First embedded payload byte (after the 12-byte header) is the opening
        # '{' of the JSON; flipping it corrupts the document.
        corrupted = _png_with_bit_flipped(stego, byte_index=HEADER_SIZE * 8)
        extracted = stego_service.extract(corrupted)
        assert extracted != original

        # Parsing or decryption must fail — corrupted plaintext is never returned.
        try:
            restored = EncryptedPayload.from_bytes(extracted)
        except ValueError:
            return  # malformed JSON failed closed at parse time
        with pytest.raises(EncryptionError):
            encryption_service.decrypt(restored, PASSWORD)

    def test_payload_bit_flip_is_handed_to_encryption(
        self, stego_service: SteganographyService, png_image: bytes
    ) -> None:
        """Payload-region corruption is not masked here; the stego layer honors
        the recorded boundary and returns modified bytes, which the encryption
        layer (GCM) is responsible for rejecting."""
        payload = _ALL_BYTES * 4
        stego = stego_service.embed(image=png_image, payload=payload)
        corrupted = _png_with_bit_flipped(stego, byte_index=HEADER_SIZE * 8 + 8)
        extracted = stego_service.extract(corrupted)
        assert extracted != payload  # corruption propagated
        assert len(extracted) == len(payload)  # boundary/length still honored
