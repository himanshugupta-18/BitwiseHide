"""
Local model registry (Phase 2.8.5).

A filesystem-backed registry for model artifacts. Intentionally simple:
no databases, no cloud services, no external infrastructure.

Each registered model lives in its own directory under the registry root:
  <registry_root>/<model_name>/<version>/

Version identity is independent from filename; the registry uses explicit
model name + version keys to prevent accidental overwrites.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai.artifact import ArtifactError, ModelArtifact

if TYPE_CHECKING:
    from ai.cnn import SuitabilityCNN


class RegistryError(Exception):
    """Raised for registry operation errors (not artifact validation errors)."""


@dataclass(frozen=True)
class RegistryEntry:
    """A registered model entry in the registry."""

    #: Model name (user-supplied, e.g., "suitability_cnn").
    model_name: str
    #: Version identifier (user-supplied, e.g., "v1", "2024-01-15-exp3").
    version: str
    #: Artifact ID from the artifact metadata.
    artifact_id: str
    #: Path to the artifact directory.
    path: Path
    #: Creation timestamp from artifact metadata (ISO 8601).
    created_at: str
    #: Optional description from artifact metadata.
    description: str = ""


class ModelRegistry:
    """
    Filesystem-backed local model registry.

    Directory structure:
      <registry_root>/
        <model_name>/
          <version>/
            metadata.json
            weights.npz

    Usage:
        registry = ModelRegistry(Path("models"))
        entry = registry.register(artifact, model_name="suitability_cnn", version="v1")
        artifact = registry.load("suitability_cnn", "v1")
    """

    def __init__(self, root: Path) -> None:
        """
        Initialize the registry at `root`.

        The root directory is created if it doesn't exist.
        """
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Registry root directory."""
        return self._root

    def _model_dir(self, model_name: str) -> Path:
        """Get the directory for a model name."""
        return self._root / model_name

    def _version_dir(self, model_name: str, version: str) -> Path:
        """Get the directory for a specific model version."""
        return self._model_dir(model_name) / version

    def _validate_name(self, name: str, field: str) -> None:
        """Validate that a name/version is a valid directory name."""
        if not name or not name.strip():
            raise RegistryError(f"{field} cannot be empty")
        # Prevent path traversal and problematic characters
        if any(c in name for c in ("/", "\\", ":", "*", "?", '"', "<", ">", "|")):
            raise RegistryError(f"{field} contains invalid characters: {name}")
        if name in (".", ".."):
            raise RegistryError(f"{field} cannot be '.' or '..'")

    def register(
        self,
        artifact: ModelArtifact,
        model_name: str,
        version: str,
        *,
        overwrite: bool = False,
        description: str = "",
    ) -> RegistryEntry:
        """
        Register an artifact in the registry.

        Args:
            artifact: The ModelArtifact to register.
            model_name: Name for the model (e.g., "suitability_cnn").
            version: Version identifier (e.g., "v1", "exp-2024-01-15").
            overwrite: If True, replace existing version; if False (default),
                raise RegistryError on duplicate.
            description: Optional description to store with the registry entry
                (overrides artifact's internal description).

        Returns:
            RegistryEntry with registration details.

        Raises:
            RegistryError: If model_name/version invalid, or version exists
                and overwrite=False.
        """
        self._validate_name(model_name, "model_name")
        self._validate_name(version, "version")

        version_dir = self._version_dir(model_name, version)
        if version_dir.exists() and not overwrite:
            raise RegistryError(
                f"Model '{model_name}' version '{version}' already exists. "
                f"Use overwrite=True to replace."
            )

        # Save artifact to the version directory
        artifact.save(version_dir)

        # Use provided description or fall back to artifact's internal description
        entry_description = description if description else artifact.metadata.description

        return RegistryEntry(
            model_name=model_name,
            version=version,
            artifact_id=artifact.metadata.artifact_id,
            path=version_dir,
            created_at=artifact.metadata.created_at,
            description=entry_description,
        )

    def load(self, model_name: str, version: str) -> ModelArtifact:
        """
        Load a registered artifact from the registry.

        Args:
            model_name: Model name.
            version: Version identifier.

        Returns:
            The loaded ModelArtifact (validated on load).

        Raises:
            RegistryError: If model/version not found.
            ArtifactError: If artifact validation fails.
        """
        self._validate_name(model_name, "model_name")
        self._validate_name(version, "version")

        version_dir = self._version_dir(model_name, version)
        if not version_dir.is_dir():
            raise RegistryError(f"Model '{model_name}' version '{version}' not found in registry.")

        return ModelArtifact.load(version_dir)

    def load_model(self, model_name: str, version: str) -> SuitabilityCNN:
        """
        Load and reconstruct the SuitabilityCNN from a registered artifact.

        Convenience method combining load() + reconstruct_model().

        Raises:
            RegistryError: If model/version not found.
            ArtifactError: If artifact validation fails.
        """
        artifact = self.load(model_name, version)
        return artifact.reconstruct_model()

    def exists(self, model_name: str, version: str) -> bool:
        """Check if a model version is registered."""
        self._validate_name(model_name, "model_name")
        self._validate_name(version, "version")
        return self._version_dir(model_name, version).is_dir()

    def list_models(self) -> list[str]:
        """List all registered model names."""
        models = []
        for entry in self._root.iterdir():
            if entry.is_dir():
                models.append(entry.name)
        return sorted(models)

    def list_versions(self, model_name: str) -> list[str]:
        """List all versions for a given model name."""
        self._validate_name(model_name, "model_name")
        model_dir = self._model_dir(model_name)
        if not model_dir.is_dir():
            return []
        versions = []
        for entry in model_dir.iterdir():
            if entry.is_dir():
                versions.append(entry.name)
        return sorted(versions)

    def list_entries(self) -> list[RegistryEntry]:
        """List all registered entries with metadata."""
        entries = []
        for model_name in self.list_models():
            for version in self.list_versions(model_name):
                try:
                    entry = self._entry_from_dir(model_name, version)
                    entries.append(entry)
                except (ArtifactError, RegistryError):
                    # Skip corrupted entries but don't fail the whole listing
                    continue
        return entries

    def _entry_from_dir(self, model_name: str, version: str) -> RegistryEntry:
        """Load a RegistryEntry from a version directory without full artifact validation."""
        version_dir = self._version_dir(model_name, version)
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.is_file():
            raise RegistryError(f"Missing metadata.json in {version_dir}")

        import json

        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"Invalid metadata.json in {version_dir}: {exc}") from exc

        return RegistryEntry(
            model_name=model_name,
            version=version,
            artifact_id=data.get("artifact_id", ""),
            path=version_dir,
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )

    def delete(self, model_name: str, version: str) -> None:
        """
        Delete a registered model version.

        Args:
            model_name: Model name.
            version: Version identifier.

        Raises:
            RegistryError: If model/version not found.
        """
        self._validate_name(model_name, "model_name")
        self._validate_name(version, "version")

        version_dir = self._version_dir(model_name, version)
        if not version_dir.is_dir():
            raise RegistryError(f"Model '{model_name}' version '{version}' not found.")

        import shutil

        shutil.rmtree(version_dir)

        # Clean up empty model directory
        model_dir = self._model_dir(model_name)
        with contextlib.suppress(OSError):
            model_dir.rmdir()

    def delete_model(self, model_name: str) -> None:
        """
        Delete all versions of a model.

        Args:
            model_name: Model name.

        Raises:
            RegistryError: If model not found.
        """
        self._validate_name(model_name, "model_name")
        model_dir = self._model_dir(model_name)
        if not model_dir.is_dir():
            raise RegistryError(f"Model '{model_name}' not found.")

        import shutil

        shutil.rmtree(model_dir)
