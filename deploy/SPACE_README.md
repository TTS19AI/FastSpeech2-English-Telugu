---
title: English Telugu FastSpeech2
emoji: 🔊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# English and Telugu Text-to-Speech

Live-text Gradio demo for https://github.com/TTS19AI/FastSpeech2-English-Telugu

Selected models: English 12000, Telugu 8000, frozen Universal HiFi-GAN.
The app retrieves the pinned project snapshot and verifies all three model hashes.
First generation can take several minutes while downloading models.
This initial Docker configuration uses CPU; response time has not been measured.

Use short sentences. Spell numbers, dates and times as words. Telugu input must
use Telugu script. Unsupported pronunciation phones are rejected explicitly.
English synthesis may remain partly noisy, as documented for the selected model.

The deployment addon has offline validation tests. A full hosted model generation
and audio listening check must still be completed before considering the demo verified.

Prior work: FastSpeech 2 (Ren et al., ICLR 2021), ming024/FastSpeech2 reference
implementation, and HiFi-GAN (Kong et al., NeurIPS 2020). Source licenses are
retained in the pinned project snapshot. This addon does not retrain weights.
