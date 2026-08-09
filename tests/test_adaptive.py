"""
Adaptive steganography analysis unit tests.

Covers Phase 2.5 behavior:
- deterministic output: same image + same weights -> identical map/ranking
- valid image handling across RGB, grayscale, and RGBA modes
- RGB and pre-grayscaled input produce identical scores (both analyze the
  "L" conversion internally)
- output dimensions always match the input image
- scores are bounded in [0.0, 1.0] and ranking is a complete, sorted priority
- visually complex (textured) regions score higher than smooth regions, and the
  top-ranked embedding locations land in the textured region
- the input image is never modified
- invalid input fails cleanly (non-PIL / zero-size -> AdaptiveAnalysisError,
  invalid weights -> ValueError)
- no ML: the analysis is a pure, deterministic scoring function

No external services are involved; all images are generated in memory.
"""

from __future__ import annotations

import pytest
from PIL import Image

from app.core.adaptive import ComplexityMap, analyze
from app.core.exceptions import AdaptiveAnalysisError

#: Flat fill used for "smooth" image regions.
_SMOOTH_GRAY = 128


def _checkerboard(size: tuple[int, int]) -> Image.Image:
    """A deterministic 0/255 checkerboard (max local contrast per 3x3 window)."""
    width, height = size
    image = Image.new("L", size)
    image.putdata([255 if (x + y) % 2 == 0 else 0 for y in range(height) for x in range(width)])
    return image


def _half_textured_half_smooth(size: tuple[int, int]) -> Image.Image:
    """Left half is a 0/255 checkerboard, right half is flat gray."""
    width, height = size
    checker = _checkerboard((width // 2, height))
    smooth = Image.new("L", (width - width // 2, height), _SMOOTH_GRAY)
    composite = Image.new("L", size)
    composite.paste(checker, (0, 0))
    composite.paste(smooth, (width // 2, 0))
    return composite


def _mean_score(result: ComplexityMap, x0: int, x1: int) -> float:
    """Mean score over a column slice [x0, x1) of the score map."""
    total = 0.0
    count = 0
    for row in result.scores:
        for score in row[x0:x1]:
            total += score
            count += 1
    return total / count


class TestDeterminism:
    """Same image + same parameters always produce the same map."""

    def test_repeated_calls_are_identical(self) -> None:
        image = _checkerboard((32, 32))
        first = analyze(image)
        second = analyze(image)
        assert first == second

    def test_independent_copies_are_identical(self) -> None:
        image = _checkerboard((32, 32))
        assert analyze(image) == analyze(image.copy())

    def test_deterministic_for_custom_weights(self) -> None:
        image = _checkerboard((32, 32))
        assert analyze(image, edge_weight=2.0, texture_weight=0.5) == analyze(
            image, edge_weight=2.0, texture_weight=0.5
        )

    def test_ranked_is_stable(self) -> None:
        image = _checkerboard((32, 32))
        assert analyze(image).ranked == analyze(image).ranked


class TestValidImages:
    """RGB, grayscale, and RGBA images all produce a usable map."""

    def test_rgb_image(self) -> None:
        result = analyze(Image.new("RGB", (16, 16), color=(10, 20, 30)))
        assert isinstance(result, ComplexityMap)
        assert result.width == 16
        assert result.height == 16

    def test_grayscale_image(self) -> None:
        result = analyze(Image.new("L", (16, 16), color=_SMOOTH_GRAY))
        assert isinstance(result, ComplexityMap)
        assert result.width == 16
        assert result.height == 16

    def test_rgba_image(self) -> None:
        result = analyze(Image.new("RGBA", (16, 16), color=(10, 20, 30, 255)))
        assert isinstance(result, ComplexityMap)
        assert result.width == 16
        assert result.height == 16


class TestRgbGrayscaleCompatibility:
    """RGB and pre-grayscaled input must score identically."""

    def test_rgb_matches_grayscale_scores(self) -> None:
        rgb = _checkerboard((32, 32)).convert("RGB")
        gray = rgb.convert("L")
        assert analyze(rgb) == analyze(gray)

    def test_rgb_matches_grayscale_ranking(self) -> None:
        rgb = _half_textured_half_smooth((64, 64)).convert("RGB")
        gray = rgb.convert("L")
        rgb_map = analyze(rgb)
        gray_map = analyze(gray)
        assert rgb_map.ranked == gray_map.ranked


class TestShape:
    """Output dimensions always match the input image."""

    def test_scores_shape_matches_input(self) -> None:
        width, height = 40, 24
        result = analyze(Image.new("RGB", (width, height), color=(5, 5, 5)))
        assert result.width == width
        assert result.height == height
        assert len(result.scores) == height
        assert all(len(row) == width for row in result.scores)

    def test_ranked_covers_every_pixel(self) -> None:
        width, height = 24, 40
        result = analyze(Image.new("L", (width, height), color=0))
        assert len(result.ranked) == width * height

    def test_ranked_coordinates_in_bounds(self) -> None:
        width, height = 20, 30
        result = analyze(Image.new("L", (width, height), color=0))
        for row, col, _score in result.ranked:
            assert 0 <= row < height
            assert 0 <= col < width


class TestScoreValidity:
    """Scores are bounded in [0.0, 1.0] and ranking is sorted and consistent."""

    def test_scores_bounded_in_unit_interval(self) -> None:
        result = analyze(_checkerboard((48, 48)))
        for row in result.scores:
            for score in row:
                assert 0.0 <= score <= 1.0

    def test_smooth_image_scores_zero(self) -> None:
        result = analyze(Image.new("L", (32, 32), color=_SMOOTH_GRAY))
        assert all(score == 0.0 for row in result.scores for score in row)

    def test_ranked_is_sorted_descending(self) -> None:
        result = analyze(_half_textured_half_smooth((64, 64)))
        scores = [entry[2] for entry in result.ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranked_matches_score_map(self) -> None:
        result = analyze(_half_textured_half_smooth((64, 64)))
        for row, col, score in result.ranked:
            assert result.scores[row][col] == pytest.approx(score)

    def test_scores_with_edge_only_weights(self) -> None:
        result = analyze(_checkerboard((32, 32)), edge_weight=1.0, texture_weight=0.0)
        assert all(0.0 <= score <= 1.0 for row in result.scores for score in row)

    def test_scores_with_texture_only_weights(self) -> None:
        result = analyze(_checkerboard((32, 32)), edge_weight=0.0, texture_weight=1.0)
        assert all(0.0 <= score <= 1.0 for row in result.scores for score in row)


class TestSuitability:
    """Visually complex regions must outrank smooth regions."""

    def test_textured_half_scores_higher_than_smooth_half(self) -> None:
        result = analyze(_half_textured_half_smooth((64, 64)))
        textured_mean = _mean_score(result, 0, 32)
        smooth_mean = _mean_score(result, 32, 64)
        assert textured_mean > smooth_mean

    def test_top_ranked_locations_are_in_textured_half(self) -> None:
        """The top quartile of embedding locations must land in the textured half."""
        width, height = 64, 64
        result = analyze(_half_textured_half_smooth((width, height)))
        top_quarter = result.ranked[: len(result.ranked) // 4]
        assert all(col < width // 2 for _row, col, _score in top_quarter)

    def test_checkerboard_beats_uniform_region(self) -> None:
        checker = analyze(_checkerboard((32, 32)))
        smooth = analyze(Image.new("L", (32, 32), color=_SMOOTH_GRAY))
        checker_mean = sum(s for row in checker.scores for s in row) / len(checker.ranked)
        smooth_mean = sum(s for row in smooth.scores for s in row) / len(smooth.ranked)
        assert checker_mean > smooth_mean


class TestNoMutation:
    """The input image is never modified by analysis."""

    def test_input_bytes_unchanged(self) -> None:
        image = _half_textured_half_smooth((64, 64)).convert("RGB")
        before = image.tobytes()
        analyze(image)
        assert image.tobytes() == before

    def test_input_mode_and_size_unchanged(self) -> None:
        image = _checkerboard((24, 16)).convert("RGB")
        analyze(image)
        assert image.mode == "RGB"
        assert image.size == (24, 16)


class TestInvalidInput:
    """Invalid input fails cleanly with the right exception."""

    def test_non_pil_image_rejected(self) -> None:
        with pytest.raises(AdaptiveAnalysisError, match="PIL image"):
            analyze("not an image")  # type: ignore[arg-type]

    def test_zero_width_rejected(self) -> None:
        with pytest.raises(AdaptiveAnalysisError, match="positive"):
            analyze(Image.new("RGB", (0, 16)))

    def test_zero_height_rejected(self) -> None:
        with pytest.raises(AdaptiveAnalysisError, match="positive"):
            analyze(Image.new("RGB", (16, 0)))

    def test_both_weights_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            analyze(Image.new("RGB", (8, 8)), edge_weight=0.0, texture_weight=0.0)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            analyze(Image.new("RGB", (8, 8)), edge_weight=-1.0, texture_weight=1.0)
