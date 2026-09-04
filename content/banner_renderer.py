"""
content/banner_renderer.py — Pillow Instagram Banner Graphic Renderer
======================================================================
Phase 5 of the Autonomous Social Media AI Agent project.

Renders 1080×1080 Instagram-format banner images using Pillow (PIL).
Each banner combines a hook headline, caption snippet, niche label, and
style-specific colour palette + decorative accent element.

STYLE_PALETTES maps every value in core.simulator.VISUAL_STYLES to a
coherent (bg_color, text_color, accent_color) triple so any style from
the gene space can be rendered without a KeyError.

Downstream consumers:
  - Phase 6 (dashboard/app.py): calls render_from_content_dict() on the
    output of ContentGenerator.generate_from_ga_result() to produce the
    preview banner shown in the dashboard.

Usage:
    python content/banner_renderer.py
    from content.banner_renderer import BannerRenderer
"""

import os
import sys
import re
import textwrap
import random

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402


# ===========================================================================
# Module-level colour palettes — High-Contrast Dark & Modern Palettes
# ===========================================================================

STYLE_PALETTES: dict[str, dict] = {
    "Midnight Violet": {
        "bg_color":       (11,  11,  20),     # #0B0B14 deep black
        "card_surface":   (21,  21,  38),     # #151526
        "text_color":     (255, 255, 255),    # #FFFFFF
        "text_muted":     (148, 163, 184),    # #94A3B8
        "accent_color":   (139, 92,  246),    # #8B5CF6 electric violet
        "accent_secondary": (192, 132, 252),  # #C084FC soft lilac
        "font_weight":    "bold",
    },
    "Cyberpunk Cyan": {
        "bg_color":       (8,   14,  26),     # #080E1A
        "card_surface":   (15,  29,  48),     # #0F1D30
        "text_color":     (248, 250, 252),    # #F8FAFC
        "text_muted":     (148, 163, 184),    # #94A3B8
        "accent_color":   (6,   182, 212),    # #06B6D4 neon cyan
        "accent_secondary": (56, 189, 248),   # #38BDF8 sky glow
        "font_weight":    "bold",
    },
    # Replaced dull dark green with high-end Sunset Amber Gold (as in Taskora screenshot)
    "Emerald Luxe": {
        "bg_color":       (16,  14,  18),     # Deep espresso shadow
        "card_surface":   (28,  24,  30),     # Elevated card surface
        "text_color":     (255, 255, 255),    # #FFFFFF
        "text_muted":     (180, 170, 160),    # Warm silver
        "accent_color":   (205, 138, 90),     # Refined Gold Amber
        "accent_secondary": (225, 160, 110),  # Soft warm coral
        "font_weight":    "bold",
    },
    "Sunset Amber / Gold": {
        "bg_color":       (16,  14,  18),     # #100E12
        "card_surface":   (28,  24,  30),     # #1C181E
        "text_color":     (255, 255, 255),    # #FFFFFF
        "text_muted":     (180, 170, 160),    # #B4AAA0
        "accent_color":   (205, 138, 90),     # #CD8A5A Gold Amber
        "accent_secondary": (225, 160, 110),  # #E1A06E
        "font_weight":    "bold",
    },
    "Crimson Noir": {
        "bg_color":       (20,  10,  13),     # #140A0D
        "card_surface":   (34,  18,  23),     # #221217
        "text_color":     (255, 241, 242),    # #FFF1F2
        "text_muted":     (156, 163, 175),    # #9CA3AF
        "accent_color":   (244, 63,  94),     # #F43F5E neon rose
        "accent_secondary": (251, 113, 133),  # #FB7185 soft coral
        "font_weight":    "bold",
    },
    # ---------- Fallback for any legacy or unexpected style -------------------
    "Bold Colorful": {
        "bg_color":       (16,  14,  18),
        "card_surface":   (28,  24,  30),
        "text_color":     (255, 255, 255),
        "text_muted":     (180, 170, 160),
        "accent_color":   (205, 138, 90),
        "accent_secondary": (225, 160, 110),
        "font_weight":    "bold",
    },
    "Minimalist": {
        "bg_color":       (11,  11,  20),
        "card_surface":   (21,  21,  38),
        "text_color":     (255, 255, 255),
        "text_muted":     (148, 163, 184),
        "accent_color":   (139, 92,  246),
        "accent_secondary": (192, 132, 252),
        "font_weight":    "bold",
    },
    "_default": {
        "bg_color":       (16,  14,  18),
        "card_surface":   (28,  24,  30),
        "text_color":     (255, 255, 255),
        "text_muted":     (180, 170, 160),
        "accent_color":   (205, 138, 90),
        "accent_secondary": (225, 160, 110),
        "font_weight":    "bold",
    },
}


# ---------------------------------------------------------------------------
# Font search paths — tried in order; first success wins
# ---------------------------------------------------------------------------
_FONT_PATHS_BOLD: list[str] = [
    "arialbd.ttf",          # Windows Arial Bold
    "Arial Bold.ttf",
    "arial.ttf",            # Windows Arial (fallback)
    "DejaVuSans-Bold.ttf",  # Linux/common
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

_FONT_PATHS_REGULAR: list[str] = [
    "arial.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Attempt to load a TrueType font at the given size."""
    paths = _FONT_PATHS_BOLD if bold else _FONT_PATHS_REGULAR
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ===========================================================================
# BannerRenderer
# ===========================================================================

class BannerRenderer:
    """Renders Instagram 1080×1080 banner images using Pillow."""

    def __init__(self, width: int = 1080, height: int = 1080) -> None:
        self.width  = width
        self.height = height

        self._font_hook    = _load_font(60, bold=True)
        self._font_body    = _load_font(34, bold=False)
        self._font_niche   = _load_font(26, bold=True)
        self._font_small   = _load_font(22, bold=True)

    # ------------------------------------------------------------------
    # Text wrapping
    # ------------------------------------------------------------------

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        draw: ImageDraw.ImageDraw,
    ) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""

        def _line_width(line: str) -> float:
            try:
                return draw.textlength(line, font=font)
            except AttributeError:
                bbox = draw.textbbox((0, 0), line, font=font)
                return float(bbox[2] - bbox[0])

        for word in words:
            test = f"{current} {word}".strip()
            if _line_width(test) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines if lines else [text]

    # ------------------------------------------------------------------
    # Specific Layout Renderers
    # ------------------------------------------------------------------

    def _render_marketing_flyer(self, img, draw, max_text_w, palette, headline, sub_headline, badge_text, footer_handle):
        def sanitize(t):
            if not t: return ""
            t = re.sub(r'[✅🚀🔥⚡⭐💡🎯📌✨]', '', t).strip()
            return "".join(c for c in t if ord(c) < 1000 or c in '•“”‘’–—')
            
        bg_color = palette["bg_color"]
        card_surface = palette.get("card_surface", (28, 24, 30))
        text_color = palette["text_color"]
        text_muted = palette.get("text_muted", (180, 170, 160))
        accent_color = palette["accent_color"]
        accent2 = palette.get("accent_secondary", accent_color)
        w, h = self.width, self.height

        # Fill background
        draw.rectangle([0, 0, w, h], fill=bg_color)
        
        # Top Accent Header Line
        draw.rectangle([60, 50, w - 60, 58], fill=accent_color)
        
        # Badge Pill Top
        if not badge_text: badge_text = "TRENDING"
        badge_text = sanitize(badge_text).upper().strip()
        if badge_text:
            try: bw = draw.textlength(badge_text, font=self._font_niche)
            except: bw = draw.textbbox((0, 0), badge_text, font=self._font_niche)[2] - draw.textbbox((0, 0), badge_text, font=self._font_niche)[0]
            pill_w = bw + 60
            pill_x = (w - pill_w) / 2
            draw.rounded_rectangle([pill_x, 80, pill_x + pill_w, 135], radius=25, fill=accent_color)
            draw.text((w // 2, 107), badge_text, font=self._font_niche, fill=bg_color, anchor="mm")
        
        # Headline - multi-line with last line/words in accent color
        headline = sanitize(headline)
        hook_display = headline if len(headline) <= 120 else headline[:117] + "..."
        lines = self._wrap_text(hook_display, self._font_hook, max_text_w - 40, draw)
        y_cursor = 175
        for idx, line in enumerate(lines[:3]):
            try: lw = draw.textlength(line, font=self._font_hook)
            except: lw = draw.textbbox((0, 0), line, font=self._font_hook)[2] - draw.textbbox((0, 0), line, font=self._font_hook)[0]
            line_color = accent2 if (idx == len(lines[:3]) - 1 and len(lines[:3]) > 1) else text_color
            draw.text(((w - lw)/2, y_cursor), line, font=self._font_hook, fill=line_color)
            y_cursor += 76

        # Structured Feature Cards (Multi-line wrap fix)
        y_cursor = max(y_cursor + 35, 430)
        sub_headline = sanitize(sub_headline)
        bullets = [b.strip().lstrip('•- 1234567890.') for b in re.split(r'[\n.]', sub_headline) if len(b.strip()) > 8]
        if not bullets:
            bullets = ["Consistent High Performance", "Targeted Growth Playbook", "Scalable Autonomous System"]

        card_margin = 80
        card_w = w - (card_margin * 2)
        card_h = 95
        gap = 20

        for idx, bullet in enumerate(bullets[:3]):
            draw.rounded_rectangle([card_margin, y_cursor, card_margin + card_w, y_cursor + card_h], radius=18, fill=card_surface, outline=accent_color, width=2)
            
            # Number circle badge
            circle_cx = card_margin + 45
            circle_cy = y_cursor + (card_h // 2)
            draw.ellipse([circle_cx - 22, circle_cy - 22, circle_cx + 22, circle_cy + 22], fill=accent_color)
            draw.text((circle_cx, circle_cy), str(idx + 1), font=self._font_niche, fill=bg_color, anchor="mm")
            
            # Wrap text inside card cleanly
            card_text_w = card_w - 110
            bullet_lines = self._wrap_text(bullet, self._font_body, card_text_w, draw)
            
            text_start_y = y_cursor + (30 if len(bullet_lines) == 1 else 16)
            for bline in bullet_lines[:2]:
                draw.text((card_margin + 90, text_start_y), bline, font=self._font_body, fill=text_color)
                text_start_y += 38

            y_cursor += card_h + gap

        # Bottom Highlight Strip & Footer Pill
        perks = "FAST RESULTS   |   100% QUALITY   |   24/7 SUPPORT"
        try: pw = draw.textlength(perks, font=self._font_small)
        except: pw = draw.textbbox((0, 0), perks, font=self._font_small)[2] - draw.textbbox((0, 0), perks, font=self._font_small)[0]
        draw.rectangle([0, h - 150, w, h - 100], fill=(22, 19, 25))
        draw.text(((w - pw)/2, h - 125), perks, font=self._font_small, fill=accent_color)
        
        footer_handle = sanitize(footer_handle or "@social_ai_agent")
        if footer_handle:
            try: fw = draw.textlength(footer_handle, font=self._font_small)
            except: fw = draw.textbbox((0, 0), footer_handle, font=self._font_small)[2] - draw.textbbox((0, 0), footer_handle, font=self._font_small)[0]
            pill_w = fw + 50
            pill_x = (w - pill_w) / 2
            draw.rounded_rectangle([pill_x, h - 75, pill_x + pill_w, h - 25], radius=20, fill=accent_color)
            draw.text((w // 2, h - 50), footer_handle, font=self._font_small, fill=bg_color, anchor="mm")

    def _render_listicle_card(self, img, draw, max_text_w, palette, headline, sub_headline, footer_handle):
        self._render_marketing_flyer(img, draw, max_text_w, palette, headline, sub_headline, "PRO TIPS", footer_handle)

    def _render_quote_card(self, img, draw, max_text_w, palette, headline, footer_handle):
        bg_color = palette["bg_color"]
        text_color = palette["text_color"]
        accent_color = palette["accent_color"]
        accent2 = palette.get("accent_secondary", accent_color)
        w, h = self.width, self.height

        draw.rectangle([0, 0, w, h], fill=bg_color)
        draw.rectangle([0, 0, 12, h], fill=accent_color)
        draw.rectangle([w - 12, 0, w, h], fill=accent_color)

        quote_color = tuple(int(c * 0.15 + a * 0.85) for c, a in zip(accent2, bg_color))
        try:
            big_font = _load_font(300, bold=True)
            draw.text((80, 50), "\u201C", font=big_font, fill=quote_color)
        except Exception:
            pass

        hook_display = headline if len(headline) <= 180 else headline[:177] + "..."
        lines = self._wrap_text(hook_display, self._font_hook, max_text_w - 100, draw)
        block_h = len(lines) * 80
        y_cursor = (h - block_h) / 2
        for line in lines:
            try: lw = draw.textlength(line, font=self._font_hook)
            except: lw = draw.textbbox((0, 0), line, font=self._font_hook)[2]
            draw.text(((w - lw)/2, y_cursor), line, font=self._font_hook, fill=text_color)
            y_cursor += 80

        if footer_handle:
            author = f"-- {footer_handle}"
            try: fw = draw.textlength(author, font=self._font_niche)
            except: fw = draw.textbbox((0, 0), author, font=self._font_niche)[2]
            draw.text(((w - fw)/2, y_cursor + 50), author, font=self._font_niche, fill=accent2)

    # ------------------------------------------------------------------
    # Core render
    # ------------------------------------------------------------------

    def render_banner(
        self,
        hook: str,
        caption_snippet: str,
        visual_style: str,
        niche: str,
        save_path: str = None,
        layout_type: str = "marketing_flyer",
        headline: str = "",
        sub_headline: str = "",
        badge_text: str = "",
        footer_handle: str = "",
    ) -> str:
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, "banner_default.png")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        palette = STYLE_PALETTES.get(visual_style, STYLE_PALETTES["_default"])
        bg_color = palette["bg_color"]

        img  = Image.new("RGB", (self.width, self.height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        margin     = 80
        max_text_w = self.width - margin * 2

        # Route by layout_type
        if layout_type == "marketing_flyer" or layout_type == "bold_banner":
            self._render_marketing_flyer(img, draw, max_text_w, palette, headline or hook, sub_headline or caption_snippet, badge_text or niche, footer_handle)
        elif layout_type == "listicle_card":
            self._render_listicle_card(img, draw, max_text_w, palette, headline or hook, sub_headline or caption_snippet, footer_handle)
        elif layout_type == "quote_card":
            self._render_quote_card(img, draw, max_text_w, palette, headline or hook, footer_handle)
        else:
            self._render_marketing_flyer(img, draw, max_text_w, palette, headline or hook, sub_headline or caption_snippet, badge_text or niche, footer_handle)

        img.save(save_path, "PNG", optimize=False)
        return save_path

    # ------------------------------------------------------------------
    # Convenience: render from ContentGenerator output dict
    # ------------------------------------------------------------------

    def render_from_content_dict(
        self,
        content_dict: dict,
        save_path: str = None,
    ) -> str:
        hook          = str(content_dict.get("hook", "Your next breakthrough starts here."))
        caption       = str(content_dict.get("caption", ""))
        visual_style  = str(content_dict.get("visual_style", "Sunset Amber / Gold"))
        niche         = str(content_dict.get("niche", "General"))
        
        layout_type   = str(content_dict.get("layout_type", "marketing_flyer"))
        headline      = str(content_dict.get("headline", hook))
        sub_headline  = str(content_dict.get("sub_headline", ""))
        badge_text    = str(content_dict.get("badge_text", niche))
        footer_handle = str(content_dict.get("footer_handle", "@social_ai_agent"))

        return self.render_banner(
            hook=hook,
            caption_snippet=caption,
            visual_style=visual_style,
            niche=niche,
            save_path=save_path,
            layout_type=layout_type,
            headline=headline,
            sub_headline=sub_headline,
            badge_text=badge_text,
            footer_handle=footer_handle
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 5b — BANNER RENDERER DEMO")
    print("=" * 65)

    renderer = BannerRenderer(width=1080, height=1080)
    save_path = os.path.join(config.OUTPUT_DIR, "test_banner.png")

    result_path = renderer.render_banner(
        hook="iPhone 16: Overhyped & Underwhelming",
        caption_snippet="Battery life drops 20%. Camera AI glitches in low light. 1099 price for recycled processor.",
        visual_style="Sunset Amber / Gold",
        niche="Controversial",
        save_path=save_path,
    )

    print(f"  Banner saved to: {result_path}")
    print("=" * 65)