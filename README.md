# Submission staging copy

This copy includes the ten generated WAVs and reviewed upstream source. Model and vocoder weights are external downloads linked below. Clean-environment inference passed for all ten frozen assignment inputs. See reproducibility/CLEAN_REPRODUCTION.md. Historical corpus-selection/export and statistics-fitting notebook cells are not yet packaged as standalone commands.

# FastSpeech 2: English and Telugu

Two independently trained acoustic models with a shared, pretrained Universal HiFi-GAN vocoder. This documentation package contains reviewed scripts, configuration, frozen assignment input tokens, inference metadata, loss histories, prediction arrays, plots and the technical report. It includes generated assignment audio and reference source. It excludes acoustic-model weights, dataset audio, alignment exports and the vocoder checkpoint. Integrate it with the existing project; it is not yet the complete public submission.

## Results and scope

| Model | Trained through | Selected step | Validation total loss | Listening observation |
|---|---:|---:|---:|---|
| Original English | 15000 | 12000 | 2.6313 | Five assignment sentences complete, partly noisy |
| Telugu | 10000 | 8000 | 2.0653 | Five assignment sentences clear and complete |

Listening observations are the project owner's subjective assessments, not MOS, ASR accuracy or independent listening results. Loss values are not directly comparable across languages. The English data_repair_v1 experiment is separate and is not used for the selected English checkpoint. Do not combine its statistics with the original English model.

## Files and external resources

Expected project layout: `English/`, `Telugu/`, `Results/`, `tools/`, `reproducibility/`, and `Technical_Report.pdf`.

The existing Drive base is `/content/drive/MyDrive/FastSpeech2_Assignment`; project is its `project` subfolder. Original source recovery archive is `project/reproducibility/FastSpeech2_upstream_d4e79eb52e8b.tar.gz`. Source commit: `d4e79eb52e8b01d24703b2dfc0385544092958f3` from https://github.com/ming024/FastSpeech2.

Required external resources, currently in the owner's Drive:

- `English/checkpoints/12000.pth.tar` and `Telugu/checkpoints/8000.pth.tar` beneath project. Check hashes in `reproducibility/selected_checkpoints.json`.
- `vocoder/UNIVERSAL_V1/g_02500000` and `config.json` beneath the base. Generator SHA-256: `771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60`.
- Full processed archives beneath base `processed_data/English/english_processed.tar.gz` and `processed_data/Telugu/telugu_processed.tar.gz` for training. Inference only needs the supplied `inference_metadata/stats.json` and `speakers.json` at the configured processed paths.
- The ten final WAVs are included in the `Results/{English,Telugu}/test_generated_audios/` folders. Preserve them when merging this package.

The shared model-download folder is listed in the Model downloads section. Add accessible model/vocoder links and verify them from an account without owner privileges before public submission. Do not commit dataset audio or third-party weights without reviewing applicable terms. The original dataset license review, particularly Telugu source terms, remains incomplete.

## Environment

Use Python 3.13 and the existing CUDA-enabled PyTorch 2.11.0 environment on a T4, as observed in the completed runs. `requirements.txt` records the runtime dependencies; it is not a guarantee that a fresh installation reproduces the historical environment bit for bit. Preserve CUDA-enabled PyTorch when already present. Do not replace it with a CPU build. `requirements-preprocessing.txt` records additional feature extraction/frontend packages; MFA is installed in its own environment, not this pip environment.

## Inference with the current Colab project

Restore the pinned upstream repository separately for each language and overlay shared patches, then the language-specific patches. Put the official generator at `<repo>/hifigan/generator_universal.pth.tar` and its matching configuration at `<repo>/hifigan/config.json`. For Telugu, the patched `text/symbols.py` with 405 symbols is mandatory. English retains 360 symbols.

Copy `English/inference_metadata/{stats.json,speakers.json}` to `/content/fastspeech2_features/english_processed/`; copy Telugu equivalents to `/content/fastspeech2_features/telugu_processed/`. Do not put English repair statistics here. Confirm model checkpoint files exist at the configured paths and match the selected hashes.

Run these commands from a Colab `%%bash` cell after restoration:

```bash
cd /content/FastSpeech2_reference
python /content/drive/MyDrive/FastSpeech2_Assignment/project/English/inference.py --device cuda --ids E1 E2a E2b E3 E4
```

```bash
cd /content/FastSpeech2_telugu
python /content/drive/MyDrive/FastSpeech2_Assignment/project/Telugu/inference.py --device cuda --ids T1 T2a T2b T3 T4
```

Working directory matters: the saved vocoder loader resolves `hifigan/config.json` and weights relative to the current directory. `--repo` only sets Python imports. Use `--device cpu` for intentional CPU inference. Frozen token inference does not run eSpeak or g2p-en.

Scripts deliberately refuse to overwrite existing WAV, JSON or NPZ outputs. These commands therefore stop if final results are already present. To reproduce independently, first make a separate project copy, change its YAML paths and both selected checkpoint path entries consistently, and use separate results folders. Checkpoint hashes must remain unchanged. Do not delete the original final results.

## Training and feature pipeline

Acoustic target settings: 22050 Hz, 80 mels, FFT/window 1024, hop 256, 0-8000 Hz mel band. Pitch and energy are phoneme-level. Dataset-specific means, standard deviations and bin bounds use training records only; validation uses the frozen training fit without clipping.

`<language>/config.yaml` is the manifest-driven raw extraction configuration. `<language>/config/preprocess.yaml` is the model/processed-data configuration. They are different inputs. The inherited `corpus_path` and English raw `val_size: 512` are unused by the custom manifest-driven extraction; the actual split counts come from manifests, not a new random upstream split.

Historical raw feature commands, after restoring corpus WAV/LAB files and audited TextGrid paths:

```bash
python /content/drive/MyDrive/FastSpeech2_Assignment/project/English/preprocess.py --config /content/drive/MyDrive/FastSpeech2_Assignment/project/English/config.yaml --split train
python /content/drive/MyDrive/FastSpeech2_Assignment/project/English/preprocess.py --config /content/drive/MyDrive/FastSpeech2_Assignment/project/English/config.yaml --split validation
python /content/drive/MyDrive/FastSpeech2_Assignment/project/English/normalize.py --config /content/drive/MyDrive/FastSpeech2_Assignment/project/English/config.yaml
```

Replace English with Telugu for the equivalent Telugu commands. Saved training-only statistics are supplied to normalize.py; it does not itself fit new statistics. Do not refit using validation. Raw extraction rejects clips with unsuitable WORLD pitch output; use the saved successful-record manifests and original extraction reports to reproduce the final 3135/626 English and 1877/357 Telugu counts.

The historical corpus selection/export/alignment and statistics-fitting notebook cells are not included as standalone CLIs in the uploaded source. The input corpus WAV/LABs, split manifests and TextGrids are also external. Full raw-to-model reproduction therefore still requires those saved resources or the original notebook. Do not describe this package alone as an end-to-end raw-data pipeline.

Historical training segments, not instructions to overwrite current checkpoints:

- English: 0->1000; 1000->5000; 5000->6000 (interrupted run, retain only records <=6000); 6000->10000; 10000->15000.
- Telugu: 0->1000; 1000->5000; 5000->10000.

Example of the validated wrapper interface: `python <project>/English/train.py --restore_step 10000 --stop_step 15000 --seed 42`. Telugu has the same interface. Use a separate experiment/config/output root for new training. The wrappers refuse to resume from an older checkpoint when a newer one already exists. Create `logs/train` and `logs/val` before starting; a previous missing-directory error interrupted English training. Model and Adam state are restored, but RNG and data-loader position are not restored, so resumed runs are not exact stochastic continuations.

## Frontends and plots

English preprocessing expands the specified time/date in E4 and explicitly selects the dictionary verb pronunciation for `conduct` in E2b. This does not establish learned context-sensitive heteronym disambiguation. E3 retains the supplied split spelling `rhyth  m`; Telugu T3 preserves its supplied original two spaces while normalization collapses whitespace and retains split tokens. Telugu uses wordwise eSpeak NG 1.51, stress marks removed, then a fixed 45-phone mapping. Unknown phones must fail; no silent dropping or symbol remapping is allowed.

The final English E2b token record ends `M EH1 TH AH0`, with no final D in `method`; report this as a frontend pronunciation limitation. No final audio or tokens were silently corrected during review. Listening completeness is not word-level correctness.

```bash
python /content/drive/MyDrive/FastSpeech2_Assignment/project/tools/plot_variance.py --language English
python /content/drive/MyDrive/FastSpeech2_Assignment/project/tools/plot_variance.py --language Telugu
```

These commands require WAVs as well as arrays and overwrite plot PNGs and the summary JSON. Preserve reviewed summaries before rerunning. Pitch is the inverse-normalized interpolated target representation, not a measured voiced/unvoiced contour from generated audio. Energy is raw STFT scale, not dB. Telugu plot labels are model aliases; `Telugu/lexicon/phone_mapping.json` gives IPA correspondences.

## Submission status

Report, code review, plots and numerical checks are complete for the supplied evidence. Audio and checkpoint generation occurred in the owner's Colab, not in this review environment. Remaining integration tasks: merge documentation without replacing existing outputs; include final WAVs; provide accessible selected-weight links; include upstream code/license and missing historical preprocessing resources; validate installation and inference in a clean environment; then publish the final repository. No MOS, ASR score, word error rate, perceptual improvement percentage or universal pronunciation accuracy is claimed.

<!-- MODEL_DOWNLOADS_START -->
## Model downloads

[Download the selected models and vocoder](https://drive.google.com/drive/folders/1jEcF1coy7FWSD_81xOOIyWJd6DDax3jY)

The project owner confirmed that this folder opens without login.

| Resource | Downloaded file must be placed at |
|---|---|
| English checkpoint, step 12000 | `English/checkpoints/12000.pth.tar` |
| Telugu checkpoint, step 8000 | `Telugu/checkpoints/8000.pth.tar` |
| Universal HiFi-GAN generator | `<reference_repo>/hifigan/generator_universal.pth.tar` |

If Drive copies have a `Copy of` prefix, rename the downloaded files
to the destination names above. Keep original training files unchanged.

Verify acoustic-model SHA-256 hashes against
`reproducibility/selected_checkpoints.json`.

Expected vocoder SHA-256:
`771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60`

Use the matching HiFi-GAN configuration and the supplied language-specific
normalization metadata. These model downloads do not include datasets.
<!-- MODEL_DOWNLOADS_END -->


<!-- CLEAN_INFERENCE_VERIFIED -->
## Verified clean inference

Public downloads and SHA-256 checks passed. Both selected models generated
all ten assignment sentences using packaged source in a fresh Python
environment.

Use `MPLBACKEND=Agg` when launching inference.

See [verification addendum](reproducibility/CLEAN_REPRODUCTION.md)
and [exact environment lock](requirements-clean-inference.lock.txt).
Earlier pending-status statements describe the pre-verification snapshot.
Full historical raw-data preprocessing reproduction remains a separate scope.


<!-- NOTEBOOK_SOURCE_ADDED -->
## Historical notebook source

The reviewed Colab export is included in `notebooks/colab_history.py`.
See `notebooks/README.md` and `notebooks/stage_index.csv` for coverage.

It contains data-selection, alignment-export, statistics-fitting,
training, recovery and clean-inference source. Earlier statements about
missing notebook source describe the pre-integration snapshot.

The historical export has an execution guard. Do not run the entire file.
Full raw-data-to-training replay has not been independently verified.
The ten-sentence clean inference test passed separately.
