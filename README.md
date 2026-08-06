# Project Apollo

A quantitative research platform for discovering and validating football forecasting signals.

Apollo does not predict match outcomes. It answers a harder question:
**"Which signals actually contain predictive power, and which are noise?"**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download raw data (requires internet)
python -m core.data_loader download

# Process into standardized format
python -m core.data_loader process

# Run first experiment
python -m experiments.EXP_001_form_bias.run
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design.

## Hypotheses

See [HYPOTHESES.md](HYPOTHESES.md) for the catalog of 25 research hypotheses.

## Principles

1. Every signal must earn its place through evidence.
2. All results measured against bookmaker baseline — not naive baselines.
3. Negative results are documented and valued.
4. No hypothesis is accepted without surviving multiple testing correction, cross-league validation, and hold-out season confirmation.
5. We build knowledge, not picks.
