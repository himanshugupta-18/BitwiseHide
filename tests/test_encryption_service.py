"""
Encryption service unit tests.

These tests verify ACTUAL security behavior, not just plumbing:
- repeated encryption of the same input yields different ciphertext/nonce/salt
- every class of tampering (ciphertext, nonce, salt, tag, version, KDF params)
  is rejected — decryption never returns corrupted plaintext
- malformed/unknown payloads fail closed
- a payload is self-describing: it decrypts even across different service configs
"""

from __future__ import annotations

import base64

import pytest

from app.core.crypto import (
    GCM_NONCE_SIZE,
    GCM_TAG_SIZE,
    SCRYPT_SALT_SIZE,
    validate_scrypt_params,
)
from app.core.exceptions import EncryptionError
from app.schemas.encryption import EncryptedPayload, KDFParams
from app.services.encryption_service import EncryptionService

PASSWORD = "correct-horse-battery-staple"  # noqa: S105 — test fixture, not a real secret
WRONG_PASSWORD = "definitely-not-the-password"  # noqa: S105


@pytest.fixture
def service() -> EncryptionService:
    return EncryptionService()


def _flip_b64(value: str) -> str:
    """Return the base64 string with one bit flipped in its decoded bytes."""
    raw = bytearray(base64.b64decode(value))
    if not raw:
        msg = "cannot tamper with an empty field"
        raise ValueError(msg)
    raw[0] ^= 0x01
    return base64.b64encode(bytes(raw)).decode("ascii")


class TestEncrypt:
    """Encryption output structure and uniqueness."""

    def test_encrypt_returns_versioned_payload(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="supersecret", password=PASSWORD)
        assert isinstance(payload, EncryptedPayload)
        assert payload.version == 1
        assert payload.kdf.name == "scrypt"
        # Ciphertext must not equal the plaintext.
        assert payload.ciphertext != base64.b64encode(b"supersecret").decode("ascii")
        # Documented format sizes hold (salt 16, nonce 12, tag 16 bytes).
        assert len(base64.b64decode(payload.salt)) == SCRYPT_SALT_SIZE
        assert len(base64.b64decode(payload.nonce)) == GCM_NONCE_SIZE
        assert len(base64.b64decode(payload.tag)) == GCM_TAG_SIZE

    def test_repeated_encryption_is_unique(self, service: EncryptionService) -> None:
        """Same password + plaintext must never produce the same payload."""
        first = service.encrypt(plaintext="same-secret", password=PASSWORD)
        second = service.encrypt(plaintext="same-secret", password=PASSWORD)
        # Fresh salt → fresh key; fresh nonce → never reused with that key.
        assert first.salt != second.salt
        assert first.nonce != second.nonce
        assert first.ciphertext != second.ciphertext
        assert first.tag != second.tag


class TestRoundTrip:
    """Encrypt/decrypt round trips."""

    def test_round_trip(self, service: EncryptionService) -> None:
        plaintext = "hunter2-secret-value"
        payload = service.encrypt(plaintext=plaintext, password=PASSWORD)
        assert service.decrypt(payload, PASSWORD) == plaintext

    def test_round_trip_unicode(self, service: EncryptionService) -> None:
        plaintext = "pässwörd 秘密 🔐"
        payload = service.encrypt(plaintext=plaintext, password=PASSWORD)
        assert service.decrypt(payload, PASSWORD) == plaintext

    def test_round_trip_empty_plaintext(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="", password=PASSWORD)
        assert service.decrypt(payload, PASSWORD) == ""

    def test_round_trip_all_payload_input_forms(self, service: EncryptionService) -> None:
        """Decrypt must accept model, bytes (embedding form), JSON, and dict."""
        plaintext = "serialized-payload"
        payload = service.encrypt(plaintext=plaintext, password=PASSWORD)
        assert service.decrypt(payload.to_bytes(), PASSWORD) == plaintext
        assert service.decrypt(payload.model_dump_json(), PASSWORD) == plaintext
        assert service.decrypt(payload.model_dump(), PASSWORD) == plaintext

    def test_payload_is_self_describing_across_configs(self) -> None:
        """A payload records its KDF params, so it must decrypt even when the
        consuming service is configured with different (still valid) params."""
        producer = EncryptionService(scrypt_n=2**14, scrypt_r=8, scrypt_p=1)
        consumer = EncryptionService(scrypt_n=2**15, scrypt_r=8, scrypt_p=1)
        payload = producer.encrypt(plaintext="cross-config", password=PASSWORD)
        assert consumer.decrypt(payload, PASSWORD) == "cross-config"


class TestTamperDetection:
    """Every modification class must fail closed with EncryptionError."""

    def test_decrypt_wrong_password(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        with pytest.raises(EncryptionError):
            service.decrypt(payload, WRONG_PASSWORD)

    def test_decrypt_empty_password(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        with pytest.raises(EncryptionError):
            service.decrypt(payload, "")

    def test_modified_ciphertext_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"ciphertext": _flip_b64(payload.ciphertext)})
        with pytest.raises(EncryptionError):
            service.decrypt(tampered, PASSWORD)

    def test_modified_nonce_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"nonce": _flip_b64(payload.nonce)})
        with pytest.raises(EncryptionError):
            service.decrypt(tampered, PASSWORD)

    def test_modified_salt_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"salt": _flip_b64(payload.salt)})
        with pytest.raises(EncryptionError):
            service.decrypt(tampered, PASSWORD)

    def test_modified_tag_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"tag": _flip_b64(payload.tag)})
        with pytest.raises(EncryptionError):
            service.decrypt(tampered, PASSWORD)

    def test_modified_version_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"version": 99})
        with pytest.raises(EncryptionError, match="Unsupported payload version"):
            service.decrypt(tampered, PASSWORD)

    def test_modified_kdf_params_rejected(self, service: EncryptionService) -> None:
        """A within-bounds param change derives a different key → tag failure."""
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"kdf": KDFParams(name="scrypt", n=2**15, r=8, p=1)})
        with pytest.raises(EncryptionError):
            service.decrypt(tampered, PASSWORD)

    def test_malicious_kdf_params_rejected_before_derivation(
        self, service: EncryptionService
    ) -> None:
        """An out-of-bounds (hostile) `n` is rejected by the bounds check, not
        by attempting an expensive — or impossible — derivation."""
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"kdf": KDFParams(name="scrypt", n=2**30, r=8, p=1)})
        with pytest.raises(EncryptionError, match="invalid KDF parameters"):
            service.decrypt(tampered, PASSWORD)

    def test_modified_kdf_name_rejected(self, service: EncryptionService) -> None:
        """An unknown KDF name fails schema validation (fail closed at parse)."""
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        data = payload.model_dump()
        data["kdf"]["name"] = "scrypt2"
        with pytest.raises(EncryptionError):
            service.decrypt(data, PASSWORD)


class TestMalformedPayload:
    """Malformed or unexpected payloads must fail closed."""

    def test_invalid_json_rejected(self, service: EncryptionService) -> None:
        with pytest.raises(EncryptionError, match="malformed"):
            service.decrypt(b"this is not json", PASSWORD)

    def test_missing_fields_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        data = payload.model_dump()
        del data["tag"]
        with pytest.raises(EncryptionError, match="malformed"):
            service.decrypt(data, PASSWORD)

    def test_unknown_fields_rejected(self, service: EncryptionService) -> None:
        """extra='forbid' means an unknown field invalidates the payload."""
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        data = payload.model_dump()
        data["surprise"] = "extra"
        with pytest.raises(EncryptionError, match="malformed"):
            service.decrypt(data, PASSWORD)

    def test_wrong_field_types_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        data = payload.model_dump()
        data["salt"] = 12345
        with pytest.raises(EncryptionError, match="malformed"):
            service.decrypt(data, PASSWORD)

    def test_invalid_base64_rejected(self, service: EncryptionService) -> None:
        payload = service.encrypt(plaintext="secret", password=PASSWORD)
        tampered = payload.model_copy(update={"ciphertext": "!!!not-base64!!!"})
        with pytest.raises(EncryptionError, match="invalid base64"):
            service.decrypt(tampered, PASSWORD)

    def test_unsupported_payload_type_rejected(self, service: EncryptionService) -> None:
        # Runtime-valid: an untyped caller passing a non-payload value must get a
        # clean EncryptionError rather than an unhandled exception.
        with pytest.raises(EncryptionError, match="Unsupported payload type"):
            service.decrypt(12345, PASSWORD)  # type: ignore[arg-type]


class TestScryptParamBounds:
    """core.crypto parameter validation (defense against crafted payloads)."""

    def test_default_params_are_accepted(self) -> None:
        validate_scrypt_params(n=2**14, r=8, p=1)

    def test_rejects_non_power_of_two_n(self) -> None:
        with pytest.raises(ValueError):
            validate_scrypt_params(n=2**14 + 1, r=8, p=1)

    def test_rejects_excessive_memory(self) -> None:
        # 128 * 2**17 * 16 bytes exceeds the memory cap.
        with pytest.raises(ValueError):
            validate_scrypt_params(n=2**17, r=16, p=1)

    def test_rejects_zero_r(self) -> None:
        with pytest.raises(ValueError):
            validate_scrypt_params(n=2**14, r=0, p=1)

    def test_service_constructor_rejects_bad_config(self) -> None:
        with pytest.raises(ValueError):
            EncryptionService(scrypt_n=2**14 + 1)
