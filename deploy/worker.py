"""A persistent process per language isolates different `text.symbols` modules."""
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid


def run_worker(connection, language, paths):
    try:
        os.environ.setdefault('MPLBACKEND', 'Agg')
        os.environ.setdefault('OMP_NUM_THREADS', '2')
        repo = Path(paths['repository'])
        root = Path(paths['project'])
        sys.path.insert(0, str(repo))
        import numpy as np
        import soundfile as sf
        import torch
        import yaml
        from model import FastSpeech2
        from text.symbols import symbols
        import hifigan
        from deploy.frontend import load_lexicon, english_phones, telugu_phones, encode_phones, validate_text
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(max(1, int(os.environ.get('TTS_CPU_THREADS', '2'))))
        folder = root / language
        preprocess = yaml.safe_load((folder / 'config/preprocess.yaml').read_text())
        config = yaml.safe_load((folder / 'config/model.yaml').read_text())
        metadata = folder / 'inference_metadata'
        preprocess['path']['preprocessed_path'] = str(metadata)
        speakers = json.loads((metadata / 'speakers.json').read_text())
        speaker_name = 'proxy_086' if language == 'English' else 'telugu_female'
        speaker = torch.tensor([speakers[speaker_name]], dtype=torch.long, device=device)
        model = FastSpeech2(preprocess, config).to(device)
        state = torch.load(paths['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval().requires_grad_(False)
        vc = json.loads((root / 'vocoder/UNIVERSAL_V1/config.json').read_text())
        vocoder = hifigan.Generator(hifigan.AttrDict(vc)).to(device)
        state = torch.load(paths['vocoder'], map_location='cpu', weights_only=True)
        vocoder.load_state_dict(state['generator'], strict=True)
        del state
        vocoder.eval().requires_grad_(False)
        vocoder.remove_weight_norm()
        if language == 'English':
            from g2p_en import G2p
            g2p = G2p()
            lexicon = load_lexicon(folder / 'lexicon/english_synthesis.dict')
        else:
            mapping = json.loads((folder / 'lexicon/phone_mapping.json').read_text())['ipa_to_token']
        sr = int(preprocess['preprocessing']['audio']['sampling_rate'])
        hop = int(preprocess['preprocessing']['stft']['hop_length'])
        # Verify inventory compatibility with frozen test IDs before accepting live input.
        frozen = json.loads((folder / 'manifests/test_sentences_tokenized.json').read_text())['sentences']
        for entry in frozen:
            if encode_phones(entry['phones'], symbols) != entry['token_ids']:
                raise RuntimeError('Restored symbol inventory does not match verified test token IDs.')
        connection.send({'ready': True})
    except Exception as exc:
        connection.send({'error': f'{type(exc).__name__}: {exc}'})
        return
    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        if request is None:
            break
        try:
            text = validate_text(request['text'], language)
            phones = english_phones(text, lexicon, g2p) if language == 'English' else telugu_phones(text, mapping)
            encoded = encode_phones(phones, symbols)
            tokens = torch.tensor([encoded], dtype=torch.long, device=device)
            lengths = torch.tensor([len(encoded)], dtype=torch.long, device=device)
            # Guard duration expansion before length regulation to bound memory use.
            def duration_guard(module, inputs, output):
                if not torch.isfinite(output).all().item():
                    raise ValueError('Non-finite predicted durations.')
                durations = torch.clamp(torch.round(torch.exp(output) - 1), min=0)
                if durations.max().item() > 600 or durations.sum().item() > 4000:
                    raise ValueError('Predicted audio is too long; try a shorter sentence.')
            guard = model.variance_adaptor.duration_predictor.register_forward_hook(duration_guard)
            try:
                with torch.inference_mode():
                    prediction = model(speaker, tokens, lengths, len(encoded))
                    frames = int(prediction[9][0].item())
                    if not 0 < frames <= 4000:
                        raise ValueError('Invalid predicted speech duration; try different wording.')
                    mel = prediction[1][0, :frames]
                    if not torch.isfinite(mel).all().item():
                        raise ValueError('Non-finite mel prediction.')
                    wav = vocoder(mel.T.unsqueeze(0))[0, 0].cpu().numpy()
            finally:
                guard.remove()
            if not wav.size or not np.isfinite(wav).all() or np.max(np.abs(wav)) > 1:
                raise ValueError('Generated audio failed amplitude checks; try different wording.')
            if wav.size != frames * hop:
                raise ValueError('Generated waveform length does not match mel length.')
            output = Path(tempfile.gettempdir()) / 'fastspeech2_web_audio'
            output.mkdir(exist_ok=True)
            path = output / (uuid.uuid4().hex + '.wav')
            sf.write(path, wav, sr, subtype='PCM_16')
            connection.send({'path': str(path), 'status': f'{language} | {speaker_name} | {wav.size / sr:.2f} s | {device.type}'})
        except Exception as exc:
            connection.send({'error': f'{type(exc).__name__}: {exc}'})
