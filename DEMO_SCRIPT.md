# Demo Script — Autonomous Social Media AI Agent & Strategy Optimization Dashboard

**Course:** AI Lab, Semester 7, IQRA University
**Total runtime target:** ~8-10 minutes

---

## 1. Opening (30 seconds)

"This is an autonomous Instagram-management simulation. It generates content,
simulates how an audience would engage with it, evaluates that performance
using four required lab algorithms, and automatically decides whether to
pivot its content strategy — all with zero paid APIs."

Show the folder structure briefly:
- `core/` (Simulation & Trend Radar)
- `ml/` (Naive Bayes & K-Means)
- `decision/` (Fuzzy Logic, GA & Pivot Manager)
- `content/` (LLM, Banner & Video Generators)
- `dashboard/` (Streamlit UI)

## 2. Command Line Execution (2 minutes)

"Let's look at the backend first. We can run the entire pipeline end-to-end, or just generate a Reel."

Run `python main.py --generate-reel`
- Show how the **Trend Radar** fetches a live topic.
- Show the **Genetic Algorithm (GA)** evolving the optimal content parameters.
- Show the **Content Generator** writing a Reel script using Groq LLM API.
- Show the **Reel Generator** synthesizing voiceover via gTTS and rendering a 9:16 vertical MP4 frame-by-frame with a dynamic gradient and subtitles.
- Open the generated MP4 file in `outputs/reels/` to prove it works.

## 3. The Dashboard Overview (1 minute)

Run `python main.py --launch-dashboard`
- **Simulation Overview Tab:** Show the 150 simulated posts. Explain how engagement metrics are generated.
- **Cluster Analysis (K-Means):** Explain how posts are grouped into Flop, Average, and Viral tiers without predefined labels.

## 4. Decision Engine (2 minutes)

- **Fuzzy Logic (Mamdani):** Show how sentiment and tier combine to produce a `shift_rate`.
- **Genetic Algorithm (Level 1 Pivot):** Point to the GA convergence chart. Explain how when `shift_rate > 0.55`, the GA evolves a new hook, visual style, and tone while keeping the topic unchanged.
- **Two-Tier Pivot Manager (Level 2 Pivot):** Explain that if a topic flops twice consecutively, the system abandons it entirely and pulls a fresh topic from the Trend Radar.

## 5. Content Studio & Revise Workflow (2 minutes)

- Go to the **Content Studio** page.
- Explain that the "winning chromosome" from the GA is passed to the LLM (Groq).
- Click **Generate Autonomous Post**. Show the generated caption, hook, hashtags, and rendered banner graphic.
- **Revise/Fine-Tune:** Demonstrate entering a custom instruction (e.g., "Make it shorter and more casual") and clicking the Revise button to prove the LLM is dynamically generating content, not just using templates.

## 6. Instagram Manager (1.5 minutes)

- Go to the **Instagram Manager** page.
- **Smart Scheduler:** Click "Generate Weekly Schedule" to show the automated posting timetable for Feed and Story.
- **Media Upload:** Show the visual heuristic tool that checks brightness and applies hook overlays.
- **Trending Audio Matcher:** Explain how it matches content themes to viral audio tracks to maximize reach.

## 7. Conclusion (30 seconds)

"By combining traditional ML (Naive Bayes, K-Means), classical AI (Fuzzy Logic, GA), and modern Generative AI (LLMs, TTS, Video rendering), this project demonstrates a fully autonomous AI capable of running a targeted social media strategy."

"Thank you. Any questions?"