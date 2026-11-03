# Simple deployment instructions

## Abhi GitHub par kya add karein?

1. ZIP extract karein.
2. Apne `FastSpeech2-English-Telugu` repository folder mein `app.py`, `Dockerfile`,
   `requirements-web.txt`, `DEPLOYMENT_HINGLISH.md` aur poora `deploy` folder copy karein.
3. Existing `README.md`, `requirements.txt`, English, Telugu aur Results files ko
   replace/delete na karein. Is addon mein un files ka replacement nahi hai.
4. GitHub Desktop mein summary likhein: `Add text-to-speech web demo`.
5. **Commit to main**, phir **Push origin** karein.

## Public website kaise banegi?

GitHub code ko store karega. Python model Hugging Face par chalega.

1. https://huggingface.co/new-space kholein aur login karein.
2. Name: `fastspeech2-english-telugu`. SDK: **Docker**, template: **Blank**.
3. Initial test ke liye CPU hardware select karein. Paid GPU select karna required nahi hai.
4. Space ke **Files** tab mein upload karein: `app.py`, `Dockerfile`,
   `requirements-web.txt` aur `deploy` folder, unki folder structure preserve karke.
5. Space ke root `README.md` ko `deploy/SPACE_README.md` ke content se update karein.
   **GitHub wale original README ko change karne ki zaroorat nahi.**
6. Build complete hone dein. Error ho to Build/Runtime log ka exact last error share karein.
7. App tab kholein, ek short sentence likhein aur **Generate speech** click karein.
   First request source aur lagbhag 1 GB model files download/load karti hai, isliye time lagega.
8. Audio play karke check karein. Telugu example app mein provided hai.

## Supported live input

- English letters and punctuation, or Telugu script and punctuation.
- Maximum 250 characters and 220 pronunciation tokens.
- Numbers/dates/time ko words mein likhein. Yeh initial version arbitrary numeral
  normalization implement nahi karta.
- The pronunciation frontend uses the trained inventory. It rejects unsupported
  phones rather than silently discarding them. New-word pronunciation is not guaranteed.
- Live text is not restricted to the original five test sentences.
- Existing final submission WAVs are never overwritten; each request has a unique WAV.

## Models aur source kahan se aate hain?

Pinned source: project commit `6f43348af7db1b49e554f03a70c6519f99ae69bf`.
Public weights: https://drive.google.com/drive/folders/1jEcF1coy7FWSD_81xOOIyWJd6DDax3jY

The default app downloads this verified source snapshot from GitHub; it does not
automatically track later GitHub edits. It hash-checks the original English/Telugu
checkpoints and vocoder. Google Drive quotas/public access can block downloads.
For an existing server, `TTS_WEIGHTS_DIR` may point to a local folder containing all
three weight files. Hashes determine which file is which, including renamed copies.
`TTS_PROJECT_ROOT` can point to an already available complete project source folder.
`TTS_CACHE` selects a writable cache directory.

Project source must contain `reference_source/FastSpeech2`, per-language configurations,
inference metadata, English synthesis lexicon, Telugu phone mapping and token manifests,
and `vocoder/UNIVERSAL_V1/config.json`. No training feature arrays/dataset required.

## Architecture and behavior

`app.py` builds the Gradio UI and serializes inference requests. `deploy/bootstrap.py`
downloads source/weights and applies saved compatibility patches to separate language
repositories. `deploy/worker.py` loads one model per language in isolated persistent
processes, avoiding Python `text.symbols` module collisions. A frozen vocoder converts
predicted post-PostNet mels to PCM-16 WAV at the configured sample rate.

`deploy/frontend.py` handles new text with a conservative pronunciation frontend:
English synthesis lexicon + G2P; Telugu wordwise eSpeak IPA + saved phone mapping.
This does not add contextual heteronym disambiguation or learned punctuation pauses.

The queue accepts up to eight pending requests. Text/token limits and a predicted
duration guard bound generation. Timeouts restart workers; outputs use unique filenames
and Gradio clears its output cache periodically.

## Checks completed and remaining

Completed locally: Python syntax; offline frontend/validation/hash unit tests.
Remaining: Docker dependency build, public source/model download, strict model-state
loading, actual synthesis for both languages, output listening and hosting latency.
This workspace does not have Docker, PyTorch or the model weight files, so a complete
runtime result has NOT been claimed. The hosting build/test is the next verification.

Deployment does not repair the selected English model's documented partly noisy output.
No training is started and the submitted project artifacts remain separate.

## Offline tests

Run from this addon's root (no ML packages or weights required):
```sh
python -m unittest discover -s deploy -p 'test_*.py' -v
```

## Optional local Docker run

```sh
docker build -t fastspeech2-web .
docker run --rm -p 7860:7860 fastspeech2-web
```
Open http://localhost:7860. Startup/generation errors are shown in the app and logs.

## Official deployment documentation

- https://huggingface.co/docs/hub/spaces-sdks-docker
- https://gradio.app/guides/queuing
- https://gradio.app/docs/audio
