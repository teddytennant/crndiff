Count-native independent birth-death kernel, data-driven terminal time, FFBS reverse sampling, and tilted Feynman-Kac steering from CRNDiff (https://arxiv.org/abs/2609.31149).

## Install

```
pip install -e ".[tests]"
```

## Run

```
pytest
python train.py
```

## What does not match the paper

No Heart Cell Atlas, scVI, gene-panel training, or neural posterior \(Q_\theta\). Reverse sampling uses the exact finite-grid posterior of a two-class Poisson mixture. Discriminator odds use the closed-form density ratio rather than a trained \(r_\phi\).
