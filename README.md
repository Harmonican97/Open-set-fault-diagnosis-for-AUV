# Open-Set Fault Diagnosis for Autonomous Underwater Vehicles via Prototype Learning and Adaptive Mahalanobis Gating

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Standard: IEEE](https://img.shields.io/badge/Standard-IEEE-blue.svg)](https://www.ieee.org/)

This repository contains the official PyTorch implementation of the paper: **"Open-Set Fault Diagnosis for Autonomous Underwater Vehicles via Prototype Learning and Adaptive Mahalanobis Gating"**.

## 📝 Abstract

Reliable fault diagnosis is critical for the mission safety of Autonomous Underwater Vehicles (AUVs). However, traditional closed-set methodologies struggle with unforeseen failure modes, while current open-set techniques often fail in underwater environments due to low-frequency, coupled telemetry and the difficulty of installing high-frequency vibration sensors.

**ProtoNet-MDAG** is a novel framework that bridges the gap between data-driven deep learning and deterministic physical laws. It features:
1.  **Dilated Multi-Scale Encoder**: Extracts robust features from low-frequency (approx. 100Hz) state data.
2.  **Feature Atomization**: Utilizes a Variance Penalty to compress known class manifolds, reserving space for unknown faults.
3.  **MDAG (Mahalanobis Distance Adaptive Gating)**: A dual-domain verification strategy coupling statistical metrics with physical residuals derived from **Newton-Euler dynamics**.

## 🚀 Key Features

- **Open-Set Recognition (OSR)**: Effectively rejects unknown anomalies (e.g., collision, entanglement) with a 99.06% rejection rate.
- **Low-Frequency Friendly**: Designed specifically for state telemetry (velocity, acceleration, control commands) rather than high-frequency vibration signals.
- **Robustness**: High performance under varying noise levels (down to 2dB SNR) and physical parameter uncertainties.

## 🛠️ Requirements

The code is implemented in Python 3.8+ and PyTorch.
