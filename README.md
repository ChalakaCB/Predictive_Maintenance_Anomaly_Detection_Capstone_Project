# Predictive Maintenance Anomaly Detection Capstone Project

The MetroPT-3 preprocessing pipeline is exposed through one command:

```bash
python src/preprocess.py
```

It creates the chronological train/validation/test feature splits and the
5/10/20/30-minute warning targets. The raw data and generated outputs are
read from and written to the fixed repository locations `data/raw/` and
`data/processed/`; both are local-only and excluded from Git.

For the concise pipeline description, see `docs/preprocess_document.md`. For
the feature naming and interpretation guide, see `docs/metadata.md`.
