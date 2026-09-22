# Discrete-deep-ARs.-From-Time-Series-to-LLMs

This project contains a case study evaluating the trade-offs between continuous autoregressive forecasting models (DeepAR) and discrete next-token prediction frameworks. Using the ETTh1 dataset and a discrete sequence modeling architecture (LSTM/Transformer backbone), we quantize real-valued time-series signals into discrete tokens, framing time-series forecasting as an LLM-style categorical prediction task.

The project was developed to demonstrate the bridge between classical probabilistic time-series forecasting and modern generative autoregressive modeling.

---

## Project Objective

The main goal is to build and compare time-series neural architectures that bridge the gap between continuous probabilistic modeling and discrete generative tokenization:

- **Continuous Baseline (M1):** Standard DeepAR-style autoregressive model outputting continuous Gaussian parameters $(\mu, \sigma^2)$ at each step.
- **Quantized Signal Tokenizer (M2):** Continuous-to-discrete pipeline mapping standardized real-valued series into $K=128$ uniform bin tokens.
- **Pure Discrete Next-Token Predictor (M3):** Autoregressive sequence architecture mapping past token embeddings to a categorical logit distribution over discrete bins using Cross-Entropy Loss.
- **Probabilistic Trajectory Generator (M4):** Autoregressive inference engine utilizing temperature scaling and Top-$k$ filtering to sample multi-step stochastic forecast trajectories.

---

## Methodology

The project is implemented in Python using **PyTorch** and follows a structured deep learning pipeline:

- **Dataset:** ETTh1 (Electricity Transformer Temperature) — focusing on the continuous `OT` (Oil Temperature) feature, standardized and subsampled for sequence modeling.
- **Quantization:** Bounded continuous values $x_t \in [x_{\text{min}}, x_{\text{max}}]$ are mapped into $K=128$ discrete tokens $z_t$ using uniform bin width $\Delta = (x_{\text{max}} - x_{\text{min}}) / K$. Continuous values are reconstructed via bin midpoints.
- **Embedding Layer:** Discrete token indices are mapped to continuous vector representations via a learned embedding table $\mathbf{E} \in \mathbb{R}^{K \times d_{\text{model}}}$.
- **Backbone Architecture:** Autoregressive core (LSTM or causal Transformer) processing sequence context to produce next-step bin distributions.
- **Training:** Categorical Cross-Entropy Loss over time-shifted target tokens, optimized via Adam optimizer.
- **Sampling Analysis:** Stochastic autoregressive forecast generation evaluated across different temperature parameters $\tau$ to capture multi-modal uncertainty.

---

## Results Summary

| Model | Target Output | Loss Function | Distributional Assumption | Multi-modal Support |
|---|---|---|---|---|
| M1 — Continuous Baseline | Continuous $(\mu, \sigma^2)$ | Gaussian NLL | Symmetric Gaussian | No |
| M2 — Quantized Tokenizer | Binned Tokens $z_t$ | — | Bounded Discrete | — |
| M3 — Pure Discrete Predictor | Categorical Logits $\mathbf{y}_t$ | Cross-Entropy | Non-parametric | Yes |
| M4 — Stochastic Generator | Sampled Paths | — | Arbitrary Empirical | Yes |

The discrete next-token framework successfully bypasses unimodal Gaussian assumptions, capturing arbitrary, asymmetric, and multi-modal posterior distributions over the forecast horizon.

---

## Conclusion

The experiments demonstrate that framing continuous time series as discrete next-token prediction offers a powerful alternative to traditional parametric forecasting. By replacing Gaussian likelihood outputs with discrete token embeddings and categorical cross-entropy, the model can capture complex multi-modal futures and non-standard probability distributions. While discretization introduces a resolution trade-off governed by bin density $K$, temperature-scaled multinomial sampling provides flexible control over forecast variance and trajectory diversity.

---

## Repository Contents

- `Discrete_DeepAR.ipynb`: Full annotated Jupyter notebook with quantization pipeline, model implementations, training loops, and autoregressive trajectory sampling.
- `report.pdf`: Written project report with technical formulation, mathematical derivation, and experimental evaluation.

---

## Dataset Preparation

The pipeline processes the `ETTh1.csv` benchmark dataset containing electricity transformer operational metrics.

**Option A — Local Dataset Path (recommended):** Place the downloaded `ETTh1.csv` file directly inside the `data/` directory relative to the repository root.
**Option B — Automated Download via Script:** If the local dataset is not found, the notebook includes fallback utilities to fetch `ETTh1.csv` directly from public benchmark mirrors or generate a synthetic benchmark series automatically.

See the notebook for the full dataset setup cell.

---

## Requirements

- Python 3.x
- PyTorch + torchvision
- Google Colab (recommended for GPU access and Drive integration)
- NumPy, Pandas, Matplotlib

---

## Authors

- Carla Aranda Sánchez 
- Jorge Barcia Belinchón
- Marina Juzgado Gómez-Menor
- Iván López Anca 
