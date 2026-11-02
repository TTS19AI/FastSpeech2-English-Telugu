# Clean inference verification

- Fresh Python environment: system_site_packages=False.
- GPU: Tesla T4; PyTorch 2.11.0+cu130.
- Public model downloads: anonymous download passed.
- English checkpoint 12000 and Telugu checkpoint 8000: hashes verified.
- Universal vocoder: hash verified.
- Packaged reference source and compatibility patches used.
- All five English and five Telugu inference runs passed.
- WAV format, duration consistency and finite arrays checked.
- Original results were preserved; no training was performed.
- Newly reproduced audio has not received a separate listening review.

## Required execution setting

Set `MPLBACKEND=Agg` before starting the inference subprocess.
Colab's inherited notebook backend is unavailable in the isolated environment.

## Interpretation

This verifies inference reproduction for the ten frozen assignment inputs.
It does not establish bitwise identical output, general pronunciation accuracy,
or full raw-data-to-training reproduction.

Technical_Report.pdf is the earlier review snapshot. This addendum records
the subsequent successful clean inference test.

See clean_inference_verification.json and clean_model_downloads.json for
the recorded results. Exact installed versions are in
requirements-clean-inference.lock.txt.
