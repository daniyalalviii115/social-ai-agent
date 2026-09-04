# Dashboard Running Instructions

To launch the Autonomous Social Media AI Agent Strategy Optimization Dashboard, activate your virtual environment from the root directory and run:

```bash
streamlit run dashboard/app.py
```

### Dashboard Pages
1. **Overview**: Key performance indicators and historical performance table of all 150 simulated posts.
2. **Cluster Analysis**: Scatter plot visualisations of post engagement along with a live predictive model for testing mock engagement scenarios.
3. **Fuzzy & GA Engine**: Live adjustable fuzzy logic shift rate computation and 40-generation genetic algorithm optimization control panel.
4. **Live Feed Simulation**: Replay mode to cycle through existing posts, plus a full end-to-end pipeline simulator that generates synthetic data, scores sentiment, predicts cluster tiers, checks fuzzy logic, runs GA if pivoted, and uses an LLM+Pillow to render a final post image live!
