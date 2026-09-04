"""
content/test_content_modules.py — Phase 5 Validation Test Suite
================================================================
Validates the correctness and integration of:
  - content/llm_generator.py  (ContentGenerator)
  - content/banner_renderer.py (BannerRenderer)

Run with:
    .\\venv\\Scripts\\python.exe content\\test_content_modules.py
"""

import os
import sys
import traceback
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from content.llm_generator import ContentGenerator  # noqa: E402
from content.banner_renderer import BannerRenderer, STYLE_PALETTES  # noqa: E402
from core.simulator import VISUAL_STYLES  # noqa: E402

_PASS = 0
_FAIL = 0


def _check(condition: bool, test_name: str, detail: str = "") -> None:
    """Print PASS / FAIL for a single assertion."""
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {test_name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {test_name}" + (f" — {detail}" if detail else ""))


# ===========================================================================
# SECTION 1: ContentGenerator Tests
# ===========================================================================

print()
print("=" * 65)
print("  SECTION 1: ContentGenerator Tests")
print("=" * 65)

try:
    generator = ContentGenerator()
    _check(isinstance(generator, ContentGenerator), "ContentGenerator instantiated successfully")

    # Test 1a — generate_content with sample args
    sample_args = {
        "niche": "Fitness",
        "hook_type": "Question",
        "content_tone": "Inspirational",
        "visual_style": "Bold Colorful",
    }
    result = generator.generate_content(**sample_args)

    _check(isinstance(result, dict), "generate_content returns a dict")

    required_keys = {
        "caption", "hook", "hashtags", "cta", "source",
        "niche", "hook_type", "content_tone", "visual_style"
    }
    _check(
        required_keys.issubset(set(result.keys())),
        f"result dict contains all required keys: {required_keys}",
        detail=f"actual keys: {list(result.keys())}"
    )

    # Hashtags validation
    hashtags = result.get("hashtags", [])
    _check(isinstance(hashtags, list) and 8 <= len(hashtags) <= 12, "hashtags is a list containing 8 to 12 items")
    all_hashtags_valid = all(isinstance(h, str) and h.startswith("#") for h in hashtags)
    _check(all_hashtags_valid, "every hashtag in hashtags starts with '#'")

    # Source validation
    source = result.get("source")
    _check(
        source in ("llm", "fallback_template"),
        f"source is either 'llm' or 'fallback_template' (got: '{source}')"
    )

    # Test 1b — generate_from_ga_result
    ga_sample = {
        "hook_type": "Question",
        "posting_hour": 18,
        "visual_style": "Minimalist",
        "content_tone": "Educational",
    }
    ga_result = generator.generate_from_ga_result(ga_sample, niche="Fitness")

    _check(isinstance(ga_result, dict), "generate_from_ga_result returns a dict")
    _check(
        required_keys.issubset(set(ga_result.keys())) and "posting_hour" in ga_result,
        "generate_from_ga_result returns all required keys plus 'posting_hour'",
    )
    _check(
        ga_result.get("posting_hour") == 18,
        "posting_hour in result matches input GA dict (18)",
    )

except Exception:
    _FAIL += 1
    print("  [FAIL] ContentGenerator raised an exception:")
    traceback.print_exc()


# ===========================================================================
# SECTION 2: BannerRenderer Tests
# ===========================================================================

print()
print("=" * 65)
print("  SECTION 2: BannerRenderer Tests")
print("=" * 65)

try:
    renderer = BannerRenderer(width=1080, height=1080)
    _check(isinstance(renderer, BannerRenderer), "BannerRenderer instantiated successfully")

    test_banner_path = os.path.join(config.OUTPUT_DIR, "test_banner.png")
    returned_path = renderer.render_banner(
        hook="Is this the productivity hack you've been missing?",
        caption_snippet="Discover how top performers structure their mornings for maximum output...",
        visual_style="Bold Colorful",
        niche="Productivity",
        save_path=test_banner_path,
    )

    _check(returned_path == test_banner_path, "render_banner returns expected save path")
    _check(os.path.exists(test_banner_path), f"Saved banner file exists on disk ({test_banner_path})")

    file_size = os.path.getsize(test_banner_path)
    _check(file_size > 0, f"Saved banner file size is > 0 bytes (actual: {file_size} bytes)")

    # PIL open dimensions verification
    with Image.open(test_banner_path) as img:
        width, height = img.size
        _check(
            (width, height) == (1080, 1080),
            f"Saved banner dimensions match configured width/height (1080x1080, got {width}x{height})"
        )

except Exception:
    _FAIL += 1
    print("  [FAIL] BannerRenderer raised an exception:")
    traceback.print_exc()


# ===========================================================================
# SECTION 3: VISUAL_STYLES rendering loop verification
# ===========================================================================

print()
print("=" * 65)
print("  SECTION 3: VISUAL_STYLES Rendering Loop Tests")
print("=" * 65)

try:
    all_styles_passed = True
    for idx, style in enumerate(VISUAL_STYLES):
        style_filename = f"test_banner_style_{idx}.png"
        style_save_path = os.path.join(config.OUTPUT_DIR, style_filename)

        out_path = renderer.render_banner(
            hook=f"Sample Hook for {style}",
            caption_snippet=f"Sample caption snippet testing rendering for {style} visual style.",
            visual_style=style,
            niche="Testing",
            save_path=style_save_path,
        )

        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
            all_styles_passed = False
            _check(False, f"Render style '{style}' failed to produce non-empty file")
        else:
            _check(True, f"Rendered visual style '{style}' successfully")

    _check(all_styles_passed, "All VISUAL_STYLES rendered with no exception or KeyError")

except Exception:
    _FAIL += 1
    print("  [FAIL] VISUAL_STYLES rendering loop raised an exception:")
    traceback.print_exc()


# ===========================================================================
# Final summary
# ===========================================================================

print()
print("=" * 65)
total = _PASS + _FAIL
print(f"  Tests passed : {_PASS} / {total}")
print(f"  Tests failed : {_FAIL} / {total}")
print("=" * 65)

if _FAIL == 0:
    print()
    print("  ALL PHASE 5 VALIDATION TESTS PASSED")
    print()
else:
    print()
    print(f"  {_FAIL} TEST(S) FAILED — review output above for details.")
    print()
    sys.exit(1)
