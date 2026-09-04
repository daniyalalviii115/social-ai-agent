"""
content/llm_generator.py — Free-Tier LLM Instagram Content Generator
======================================================================
Phase 5 of the Autonomous Social Media AI Agent project.
Updated in Phase 8 (Dashboard Overhaul) to support an optional
`user_instruction` free-text revision hint for the Content Studio's
Revise / Fine-Tune workflow — fully backward compatible with all
existing call sites and test suites.

Generates Instagram captions, hooks, hashtags, and CTAs via either the
Google Gemini (gemini-1.5-flash) or Groq (openai/gpt-oss-120b)
free-tier API, branching on config.LLM_PROVIDER.

When the API key is missing/invalid or the call fails after retries,
a fully-implemented template-based fallback system produces valid,
varied content so the pipeline never halts.
"""

import json
import os
import random
import re
import sys
import time
import requests
import google.generativeai as genai
from groq import Groq

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

import config
from core.simulator import NICHES


def _sanitize_for_hashtag(text: str) -> str:
    """Strips all non-alphanumeric characters, producing a clean hashtag-safe base string."""
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _clean_and_parse_json(raw_text: str) -> dict:
    """
    Safely extract and parse JSON from raw LLM output. Handles markdown code
    fences, trailing commas, smart/curly quotes, and — as a last resort —
    regex-extracts individual known fields if the JSON is malformed, so a
    single stray character in the LLM's response doesn't force a full
    fallback-to-template when most of the content is actually usable.
    """
    text = raw_text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        snippet = cleaned[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
    extracted = {}
    patterns = {
        "hook_headline": r'"hook_headline"\s*:\s*"([^"]+)"',
        "hook": r'"hook"\s*:\s*"([^"]+)"',
        "caption": r'"caption"\s*:\s*"([^"]+)"',
        "cta": r'"cta"\s*:\s*"([^"]+)"',
    }
    for key, pat in patterns.items():
        m = re.search(pat, raw_text)
        if m:
            extracted[key] = m.group(1)
    if "hook" in extracted or "hook_headline" in extracted:
        extracted.setdefault("hashtags", ["#trending", "#viral", "#insights"])
        extracted.setdefault("cta", "Follow for more updates.")
        return extracted
    raise ValueError(f"Unable to parse LLM response into valid JSON: {raw_text[:200]}")


# ===========================================================================
# ContentGenerator
# ===========================================================================

class ContentGenerator:
    """
    Free-tier LLM content generator for Instagram posts.
    """

    _CAPTION_TEMPLATES: list[str] = [
        "Ready to level up your {niche} game? The secret isn't hustle — it's "
        "having the right strategy. I've tested this approach for 90 days and "
        "the results speak for themselves. Drop a 🔥 if you want the full breakdown.",

        "Most people in {niche} are doing it wrong. Not because they lack "
        "effort, but because no one taught them the framework. Here's what the "
        "top 1% actually do differently — and you can start today. Save this "
        "post and come back to it when you're ready to commit.",

        "Unpopular opinion: grinding harder in {niche} is not the answer. "
        "Working smarter — with data, strategy, and consistency — is. I spent "
        "months learning this the hard way so you don't have to. Which tip "
        "resonates most with you? Tell me in the comments 👇",

        "The {niche} space is flooded with noise. Here's what actually moves "
        "the needle: a clear goal, a repeatable system, and the patience to "
        "let it compound. Stop chasing tactics and start building foundations. "
        "Share this with someone who needs to hear it today.",

        "Three months ago I was struggling with {niche}. Today everything has "
        "changed — not because of luck, but because of one simple mindset "
        "shift. The journey is messy but the destination is worth it. If this "
        "resonates with you, follow for more honest insights every week.",

        "What separates people who crush it in {niche} from those who don't? "
        "It's not talent. It's not resources. It's the willingness to do the "
        "boring, unglamorous work consistently. Here's exactly what that looks "
        "like in practice — no fluff, no BS.",

        "I used to think {niche} success was about big dramatic moves. I was "
        "wrong. It's about tiny, intentional steps repeated every single day. "
        "The compound effect is real and it will surprise you. Start small, "
        "stay consistent, and trust the process.",
    ]

    _HOOK_TEMPLATES: dict[str, list[str]] = {
        "Question": [
            "What if everything you knew about {niche} was holding you back?",
            "Are you making this critical mistake in {niche} right now?",
            "Why do 97% of people never see real results in {niche}?",
            "What would your {niche} journey look like if you had a real roadmap?",
            "Have you ever wondered why some people thrive in {niche} while others struggle?",
        ],
        "Bold Statement": [
            "The {niche} advice everyone gives you is completely wrong.",
            "I stopped following conventional {niche} wisdom — and it changed everything.",
            "This single {niche} principle outperforms every hack I've ever tried.",
            "The biggest {niche} lie you've been told — debunked.",
            "Most {niche} content is noise. This is the signal.",
        ],
        "Relatable Struggle": [
            "I was exhausted, overwhelmed, and ready to quit {niche}. Then this happened.",
            "For two years I did everything 'right' in {niche} and still saw nothing.",
            "Nobody talks about how hard {niche} actually is at the beginning.",
            "I wasted months in {niche} before learning this lesson.",
            "If you've ever felt like you're invisible in {niche}, this is for you.",
        ],
        "Listicle Tease": [
            "5 {niche} habits that changed my life in 30 days 👇",
            "The 3 things nobody tells you about starting in {niche}:",
            "7 {niche} shortcuts the pros use (and never share publicly):",
            "4 brutal truths about {niche} that will save you years of wasted effort:",
            "6 underrated {niche} strategies working right now:",
        ],
        "Before/After": [
            "From complete {niche} beginner to consistent results in 90 days — here's how.",
            "Before {niche}: stressed and lost. After: clear, focused, winning.",
            "My {niche} transformation started the day I stopped doing this one thing.",
            "6 months ago I had zero traction in {niche}. This is what changed.",
            "The exact shift that took my {niche} from chaos to clarity.",
        ],
        "Controversial Take": [
            "Hot take: the way we approach {niche} is fundamentally broken.",
            "Controversial opinion: most {niche} advice is actively harmful.",
            "I disagree with every {niche} 'expert' on this — and here's my proof.",
            "The {niche} strategy everyone recommends? I tried it and it failed.",
            "Nobody wants to say this about {niche}, so I will.",
        ],
        "Story Opener": [
            "It was 11pm and I was about to give up on {niche} forever.",
            "The day I got my {niche} breakthrough, I almost missed it entirely.",
            "Three years ago a mentor said one thing about {niche} that stuck with me.",
            "I remember the exact moment everything clicked with {niche}.",
            "It started as a small {niche} experiment. What happened next surprised everyone.",
        ],
        "Stat Shock": [
            "92% of people who try {niche} quit within 3 months. Here's how to be in the 8%.",
            "Studies show the #1 predictor of {niche} success isn't what you think.",
            "The average person wastes 6 months in {niche} before finding this strategy.",
            "Only 1 in 10 people in {niche} master this — and they dominate.",
            "Data shows {niche} results compound 3x faster with this one habit.",
        ],
    }

    _HASHTAG_POOL: dict[str, list[str]] = {
        "universal": [
            "#growthmindset", "#successmindset", "#dailymotivation",
            "#entrepreneurmindset", "#worksmarter", "#strategy",
            "#results", "#levelup", "#contentcreator", "#personalbranding",
            "#digitalmarketing", "#socialmediatips", "#onlinebusiness",
            "#productivity", "#motivation", "#inspiration", "#selfimprovement",
            "#goalsetting", "#mindset", "#consistency",
        ],
        "Fitness": [
            "#fitness", "#fitnessmotivation", "#workout", "#gym", "#fitlife",
            "#health", "#healthylifestyle", "#training", "#fitnessjourney",
            "#bodygoals", "#strongnotskinny", "#fitspo", "#personaltrainer",
        ],
        "Personal Finance": [
            "#personalfinance", "#financialfreedom", "#investing", "#money",
            "#wealthbuilding", "#financetips", "#savemoney", "#budgeting",
            "#stockmarket", "#financialindependence", "#moneymanagement",
            "#richhabits", "#financegoals",
        ],
        "Tech Reviews": [
            "#tech", "#technology", "#techreview", "#gadgets", "#innovation",
            "#software", "#AI", "#startup", "#coding", "#developer",
            "#techtips", "#futuretech", "#digitaltransformation",
        ],
        "Food & Recipes": [
            "#food", "#foodie", "#recipe", "#cooking", "#foodphotography",
            "#homecooking", "#eatwell", "#foodlover", "#instafood",
            "#delicious", "#healthyfood", "#mealprep", "#foodblogger",
        ],
        "Travel": [
            "#travel", "#wanderlust", "#travelgram", "#adventure", "#explore",
            "#travelblogger", "#travelphotography", "#vacation", "#roadtrip",
            "#traveltheworld", "#bucketlist", "#nomad", "#traveltips",
        ],
    }

    _CTA_TEMPLATES: list[str] = [
        "Follow for weekly {niche} tips that actually work. 🚀",
        "Save this post and share it with someone on a {niche} journey. 💪",
        "Drop your biggest {niche} challenge in the comments — I read every one.",
        "Tag a friend who needs to see this {niche} insight today. 👇",
        "Hit follow for more no-fluff {niche} content every week.",
        "What's your top {niche} tip? Share it below and let's learn together. 🌟",
        "Double-tap if this resonated, and follow for more honest {niche} content.",
    ]

    def __init__(self) -> None:
        self.provider: str = config.LLM_PROVIDER.lower().strip()

        if self.provider == "gemini":
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.client = genai.GenerativeModel("gemini-1.5-flash")
            self._model_name: str = "gemini-1.5-flash"
        else:
            self.provider = "groq"
            self.client = Groq(api_key=config.GROQ_API_KEY)
            # llama-3.3-70b-versatile was deprecated by Groq on June 17, 2026;
            # openai/gpt-oss-120b is Groq's officially recommended replacement.
            self._model_name = "openai/gpt-oss-120b"

    def _build_prompt(
        self,
        niche: str,
        hook_type: str,
        content_tone: str,
        visual_style: str,
        user_instruction: str = "",
    ) -> str:
        instruction_block = ""
        if user_instruction and user_instruction.strip():
            instruction_block = f"""

IMPORTANT — additional revision instruction from the user (apply this on top
of everything above, while still returning valid JSON in the same schema):
"{user_instruction.strip()}"
"""

        return f"""You are an expert Instagram content strategist specialising in the {niche} niche.

Your task: Generate a high-performing Instagram post using these parameters:
- Niche: {niche}
- Hook Type: {hook_type}
- Content Tone: {content_tone}
- Visual Style: {visual_style}
{instruction_block}
You MUST respond with ONLY valid JSON, no markdown code fences, no extra text, no commentary before or after.

The JSON must have exactly these keys:
{{
  "hook": "<One attention-grabbing opening line that matches the {hook_type} hook type. Max 20 words. Should stop the scroll immediately.>",
  "caption": "<The main caption body: 2-4 sentences. Tone must be {content_tone}. Authentic, specific, and valuable. No generic advice. End with engagement prompt.>",
  "hashtags": ["#hashtag1", "#hashtag2", ...],
  "cta": "<One clear, specific call-to-action sentence. Action-oriented, max 20 words.>"
}}

Requirements:
- hashtags: list of 8-12 strings, each MUST start with #, mix of niche-specific and broad reach tags
- hook: must clearly reflect the '{hook_type}' hook archetype
- caption: must feel authentic to someone passionate about {niche}, with a {content_tone} tone, suited for a {visual_style} visual aesthetic
- All values must be strings (or list of strings for hashtags)
- Do not include any text outside the JSON object

Respond with ONLY valid JSON, no markdown code fences, no extra text."""

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "gemini":
            api_key = config.GEMINI_API_KEY.strip()
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

            # Key format ke mutabiq headers aur params set karein
            if api_key.startswith("AQ."):
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                params = {}
            else:
                headers = {"Content-Type": "application/json"}
                params = {"key": api_key}

            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }

            resp = requests.post(url, headers=headers, params=params, json=payload, timeout=25)
            if resp.status_code != 200:
                raise ValueError(f"Gemini API Error {resp.status_code}: {resp.text}")

            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            completion = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=900,
            )
            return completion.choices[0].message.content

    @staticmethod
    def _strip_markdown_fences(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[: text.rfind("```")].rstrip()
        return text.strip()

    @staticmethod
    def _apply_fallback_revision(content: dict, user_instruction: str) -> dict:
        """
        Best-effort, rule-based approximation of a 'revision' when running in
        template-fallback mode (no live LLM available). This cannot truly
        rewrite text the way an LLM would, but it applies a few common,
        deterministic tweaks so the Revise / Fine-Tune button in the
        dashboard still visibly changes the output rather than silently
        no-op-ing when no API key is configured.
        """
        instr = user_instruction.lower()

        # "shorten" / "shorter" / "short" -> trim caption to its first sentence
        if any(w in instr for w in ("short", "shorten", "brief", "concise")):
            first_sentence = re.split(r"(?<=[.!?])\s+", content["caption"].strip())[0]
            content["caption"] = first_sentence

        # "casual" -> strip a couple of the more formal-sounding CTA phrasing
        if "casual" in instr:
            content["cta"] = content["cta"].replace("Hit follow for", "Follow for")

        # "educational" -> prepend a small instructional framing to the hook
        if "educational" in instr or "teach" in instr:
            if not content["hook"].lower().startswith(("here's", "here is")):
                content["hook"] = "Here's what actually works: " + content["hook"][0].lower() + content["hook"][1:]

        # "fewer hashtags" / "less hashtags"
        if "fewer hashtag" in instr or "less hashtag" in instr or "remove hashtag" in instr:
            content["hashtags"] = content["hashtags"][:5]

        return content

    def _fallback_content(
        self,
        niche: str,
        hook_type: str,
        content_tone: str,
        user_instruction: str = "",
    ) -> dict:
        caption = random.choice(self._CAPTION_TEMPLATES).format(niche=niche)

        hook_pool = self._HOOK_TEMPLATES.get(
            hook_type,
            self._HOOK_TEMPLATES["Question"],
        )
        hook = random.choice(hook_pool).format(niche=niche)

        # Sanitized base for clean hashtags
        niche_clean = _sanitize_for_hashtag(niche)
        dynamic_niche_tags = [
            f"#{niche_clean}",
            f"#{niche_clean}tips",
            f"#{niche_clean}life",
            f"#{niche_clean}community",
            "#viral", "#trending", "#explore", "#community", "#smallbusiness"
        ]

        combined_pool = list(dict.fromkeys(
            self._HASHTAG_POOL["universal"] +
            self._HASHTAG_POOL.get(niche, []) +
            dynamic_niche_tags
        ))

        combined_pool = [t if t.startswith("#") else f"#{t}" for t in combined_pool]
        hashtags = random.sample(combined_pool, k=random.randint(8, 12))
        cta = random.choice(self._CTA_TEMPLATES).format(niche=niche)

        result = {
            "caption":  caption,
            "hook":     hook,
            "hashtags": hashtags,
            "cta":      cta,
        }

        if user_instruction and user_instruction.strip():
            result = self._apply_fallback_revision(result, user_instruction)

        return result

    def generate_content(
        self,
        niche: str,
        hook_type: str,
        content_tone: str,
        visual_style: str,
        max_retries: int = 2,
        user_instruction: str = "",
    ) -> dict:
        """
        Generate a full Instagram content package.

        Parameters
        ----------
        ... (unchanged)
        user_instruction : str, optional
            Free-text revision hint from the Content Studio's
            "Revise / Fine-Tune" box (e.g. "make it more casual",
            "shorten the caption"). Empty string = standard autonomous
            generation, identical behaviour to before this parameter
            was added — fully backward compatible.
        """
        _placeholder_fragments = {
            "your_gemini_api_key_here", "your_groq_api_key_here", "placeholder", "insert"
        }
        key = (
            config.GEMINI_API_KEY if self.provider == "gemini"
            else config.GROQ_API_KEY
        )
        key_is_placeholder = (
            not key
            or len(key) < 10
            or any(frag in key for frag in _placeholder_fragments)
        )

        prompt = self._build_prompt(niche, hook_type, content_tone, visual_style, user_instruction)

        if not key_is_placeholder:
            for attempt in range(max_retries + 1):
                try:
                    raw = self._call_llm(prompt)
                    parsed = _clean_and_parse_json(raw)

                    required = {"caption", "hook", "hashtags", "cta"}
                    if not required.issubset(parsed.keys()):
                        raise ValueError(
                            f"LLM response missing keys: {required - set(parsed.keys())}"
                        )

                    hashtags = parsed.get("hashtags", [])
                    if not isinstance(hashtags, list) or len(hashtags) == 0:
                        raise ValueError("hashtags must be a non-empty list")

                    hashtags = [
                        h if h.startswith("#") else f"#{h}"
                        for h in hashtags
                        if isinstance(h, str)
                    ]
                    parsed["hashtags"] = hashtags

                    return {
                        "caption":      str(parsed.get("caption", "")),
                        "hook":         str(parsed.get("hook", "")),
                        "hashtags":     hashtags,
                        "cta":          str(parsed.get("cta", "")),
                        "source":       "llm",
                        "niche":        niche,
                        "hook_type":    hook_type,
                        "content_tone": content_tone,
                        "visual_style": visual_style,
                    }

                except Exception as exc:
                    wait = 1.5 * (attempt + 1)
                    print(
                        f"  [LLM] Attempt {attempt + 1}/{max_retries + 1} failed "
                        f"({type(exc).__name__}: {exc}). "
                        + (f"Retrying in {wait:.1f}s..." if attempt < max_retries
                           else "Falling back to template.")
                    )
                    if attempt < max_retries:
                        time.sleep(wait)

        else:
            print(
                f"  [LLM] API key for '{self.provider}' appears to be a placeholder. "
                "Using fallback template system."
            )

        fallback = self._fallback_content(niche, hook_type, content_tone, user_instruction)
        return {
            "caption":      fallback["caption"],
            "hook":         fallback["hook"],
            "hashtags":     fallback["hashtags"],
            "cta":          fallback["cta"],
            "source":       "fallback_template",
            "niche":        niche,
            "hook_type":    hook_type,
            "content_tone": content_tone,
            "visual_style": visual_style,
        }

    def generate_from_ga_result(
        self,
        ga_best_individual: dict,
        niche: str = None,
        user_instruction: str = "",
    ) -> dict:
        if niche is None:
            niche = random.choice(NICHES)

        hook_type    = ga_best_individual["hook_type"]
        posting_hour = ga_best_individual["posting_hour"]
        visual_style = ga_best_individual["visual_style"]
        content_tone = ga_best_individual["content_tone"]

        result = self.generate_content(
            niche=niche,
            hook_type=hook_type,
            content_tone=content_tone,
            visual_style=visual_style,
            user_instruction=user_instruction,
        )
        result["posting_hour"] = posting_hour
        return result

    # ------------------------------------------------------------------
    # Phase 2 — Dynamic Reel/Post Scripting
    # ------------------------------------------------------------------

    def _build_script_prompt(
        self,
        niche: str,
        hook_type: str,
        content_tone: str,
        live_topic: str,
        script_type: str,
    ) -> str:
        format_desc = (
            "a 20-30 second vertical Instagram Reel voiceover script "
            "(3-4 short, punchy bulleted lines a narrator would read aloud)"
            if script_type == "reel"
            else "a standard Instagram feed post (no voiceover script needed)"
        )

        return f"""You are an expert short-form Instagram content strategist specialising in the {niche} niche.

Your task: Write content about this specific trending topic: "{live_topic}"

Format required: {format_desc}
Hook Type: {hook_type}
Content Tone: {content_tone}

You MUST respond with ONLY valid JSON, no markdown code fences, no extra text.

The JSON must have exactly these keys:
{{
  "hook_headline": "<Attention-grabbing headline, STRICTLY under 12 words, about '{live_topic}'>",
  "voiceover_script": ["<punchline 1>", "<punchline 2>", "<punchline 3>", "<punchline 4 (optional)>"],
  "caption": "<Full Instagram caption, 2-4 sentences, {content_tone} tone, about '{live_topic}'>",
  "cta": "<One clear call-to-action sentence>",
  "hashtags": ["#hashtag1", "#hashtag2", ...]
}}

Requirements:
- voiceover_script: exactly 3 or 4 short lines, each readable aloud in 5-8 seconds (fits a 20-30s reel total)
- hashtags: 8-12 strings, each starting with #, mixing niche-specific and topic-specific tags
- Everything must be specifically about "{live_topic}", not generic {niche} content
- Do not include any text outside the JSON object

Respond with ONLY valid JSON, no markdown code fences, no extra text."""

    def _fallback_script_content(
        self,
        niche: str,
        hook_type: str,
        content_tone: str,
        live_topic: str,
    ) -> dict:
        """
        Template-based fallback for reel/post scripts when no live LLM key
        is available. Cannot write truly novel topic-aware prose, but slots
        the live_topic into generic, structurally-sound punchlines so the
        pipeline (and video_generator.py downstream) always has usable
        voiceover text to synthesize speech from.
        """
        hook_headline = f"{live_topic}: what you need to know right now"
        if len(hook_headline.split()) > 12:
            hook_headline = f"{live_topic}: what's happening"

        voiceover_script = [
            f"Everyone's talking about {live_topic} right now.",
            f"Here's why it matters for {niche.lower()}.",
            "Three things you should know before you scroll past this.",
            "Follow for more updates like this every week.",
        ]

        base_content = self._fallback_content(niche, hook_type, content_tone)
        niche_clean = _sanitize_for_hashtag(niche)
        topic_clean = _sanitize_for_hashtag(live_topic)
        topic_tags = [f"#{topic_clean}", f"#{niche_clean}news", "#trending"]
        hashtags = list(dict.fromkeys(topic_tags + base_content["hashtags"]))[:12]

        return {
            "hook_headline":     hook_headline,
            "voiceover_script":  voiceover_script,
            "caption":           base_content["caption"],
            "cta":               base_content["cta"],
            "hashtags":          hashtags,
        }

    def generate_script(
        self,
        best_individual: dict,
        live_topic: str,
        script_type: str = "reel",
        niche: str = None,
        max_retries: int = 2,
    ) -> dict:
        """
        Generate a structured Reel or Post script around a specific live
        trending topic (Phase 2 — Dynamic Scripting).

        Parameters
        ----------
        best_individual : dict — gene combo (hook_type, posting_hour,
                          visual_style, content_tone), typically the output
                          of a GA run or a pivot_manager decision.
        live_topic      : str  — the specific trending topic to script around
                          (e.g. from core.trend_radar.TrendRadar).
        script_type     : "reel" | "post"
        niche           : str, optional — defaults to a random niche.

        Returns
        -------
        dict with keys: hook_headline, voiceover_script (list[str]),
        caption, cta, hashtags, source, niche, script_type, live_topic,
        posting_hour, visual_style.
        """
        if script_type not in ("reel", "post"):
            script_type = "reel"

        if niche is None:
            niche = random.choice(NICHES)

        hook_type    = best_individual.get("hook_type", "Question")
        content_tone = best_individual.get("content_tone", "Inspirational")
        visual_style = best_individual.get("visual_style", "Bold Colorful")
        posting_hour = best_individual.get("posting_hour", 21)

        _placeholder_fragments = {
            "your_gemini_api_key_here", "your_groq_api_key_here", "placeholder", "insert"
        }
        key = config.GEMINI_API_KEY if self.provider == "gemini" else config.GROQ_API_KEY
        key_is_placeholder = (
            not key or len(key) < 10 or any(frag in key for frag in _placeholder_fragments)
        )

        if not key_is_placeholder:
            prompt = self._build_script_prompt(niche, hook_type, content_tone, live_topic, script_type)
            for attempt in range(max_retries + 1):
                try:
                    raw = self._call_llm(prompt)
                    parsed = _clean_and_parse_json(raw)

                    required = {"hook_headline", "voiceover_script", "caption", "cta", "hashtags"}
                    if not required.issubset(parsed.keys()):
                        raise ValueError(f"LLM response missing keys: {required - set(parsed.keys())}")

                    hashtags = [
                        h if h.startswith("#") else f"#{h}"
                        for h in parsed.get("hashtags", [])
                        if isinstance(h, str)
                    ]
                    voiceover = [str(line) for line in parsed.get("voiceover_script", []) if str(line).strip()]

                    return {
                        "hook_headline":    str(parsed.get("hook_headline", "")),
                        "voiceover_script": voiceover,
                        "caption":          str(parsed.get("caption", "")),
                        "cta":              str(parsed.get("cta", "")),
                        "hashtags":         hashtags,
                        "source":           "llm",
                        "niche":            niche,
                        "script_type":      script_type,
                        "live_topic":       live_topic,
                        "posting_hour":     posting_hour,
                        "visual_style":     visual_style,
                    }
                except Exception as exc:
                    wait = 1.5 * (attempt + 1)
                    print(
                        f"  [LLM] Script attempt {attempt + 1}/{max_retries + 1} failed "
                        f"({type(exc).__name__}: {exc}). "
                        + (f"Retrying in {wait:.1f}s..." if attempt < max_retries
                           else "Falling back to template.")
                    )
                    if attempt < max_retries:
                        time.sleep(wait)
        else:
            print(
                f"  [LLM] API key for '{self.provider}' appears to be a placeholder. "
                "Using fallback template script."
            )

        fallback = self._fallback_script_content(niche, hook_type, content_tone, live_topic)
        return {
            "hook_headline":    fallback["hook_headline"],
            "voiceover_script": fallback["voiceover_script"],
            "caption":          fallback["caption"],
            "cta":              fallback["cta"],
            "hashtags":         fallback["hashtags"],
            "source":           "fallback_template",
            "niche":            niche,
            "script_type":      script_type,
            "live_topic":       live_topic,
            "posting_hour":     posting_hour,
            "visual_style":     visual_style,
        }


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 5a — LLM CONTENT GENERATOR DEMO")
    print("=" * 65)

    generator = ContentGenerator()
    print(f"  Provider : {generator.provider}  |  Model: {generator._model_name}\n")

    result = generator.generate_content(
        niche="Food & Recipes",
        hook_type="Question",
        content_tone="Inspirational",
        visual_style="Bold Colorful",
    )

    print("  Generated content:")
    print(json.dumps(result, indent=2))
    print()
    print(f"  Source: '{result['source']}'")
    print("=" * 65)