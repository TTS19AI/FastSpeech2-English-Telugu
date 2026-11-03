"""Fetch the verified project snapshot and hash-verify original inference weights."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

COMMIT = '6f43348af7db1b49e554f03a70c6519f99ae69bf'
REPO = 'https://github.com/TTS19AI/FastSpeech2-English-Telugu.git'
FOLDER = 'https://drive.google.com/drive/folders/1jEcF1coy7FWSD_81xOOIyWJd6DDax3jY'
HASHES = {
    'English': 'fbf365c7c63a670d51fbd2b4e95e2dccf4597d62827e463349603af694c0a910',
    'Telugu': '93fe938dbf91cabae46e26e2da6414d92a1571a8aa568f89521ad3487f2e4fa2',
    'vocoder': '771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60',
}
CACHE = Path(os.environ.get('TTS_CACHE', str(Path(tempfile.gettempdir()) / 'fastspeech2_web')))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def project_root():
    configured = os.environ.get('TTS_PROJECT_ROOT')
    target = Path(configured) if configured else CACHE / 'project_snapshot'
    if not configured and not (target / '.ready').exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='source_', dir=CACHE))
        try:
            for args in [['git', 'init', str(staging)],
                         ['git', '-C', str(staging), 'fetch', '--depth', '1', REPO, COMMIT],
                         ['git', '-C', str(staging), 'checkout', '--detach', 'FETCH_HEAD']]:
                subprocess.run(args, check=True, timeout=300)
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
            (target / '.ready').write_text(COMMIT)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    candidates = [target, target / 'project']
    for root in candidates:
        if (root / 'English/config/model.yaml').is_file() and (root / 'Telugu/config/model.yaml').is_file():
            return root
    raise RuntimeError('Project source is incomplete: English/Telugu configuration folders missing.')


def weights():
    source = Path(os.environ.get('TTS_WEIGHTS_DIR', str(CACHE / 'weights')))
    manifest = CACHE / 'verified_weights.txt'
    # Always verify bytes before loading; downloaded names may be "Copy of ...".
    def scan():
        found = {}
        if source.exists():
            for path in source.rglob('*'):
                if path.is_file() and path.stat().st_size > 1024 * 1024:
                    digest = sha256(path)
                    for name, expected in HASHES.items():
                        if digest == expected:
                            found[name] = path.resolve()
        return found
    found = scan()
    if len(found) != 3 and not os.environ.get('TTS_WEIGHTS_DIR'):
        import gdown
        source.mkdir(parents=True, exist_ok=True)
        result = gdown.download_folder(url=os.environ.get('TTS_MODEL_FOLDER', FOLDER),
                                       output=str(source), quiet=False, remaining_ok=True)
        if result is None:
            raise RuntimeError('Model download failed. Check public Drive access or use TTS_WEIGHTS_DIR.')
        found = scan()
    missing = sorted(set(HASHES) - set(found))
    if missing:
        raise RuntimeError(f'Missing or hash-mismatched model files: {missing}. Check public model downloads.')
    return found


def prepare():
    root = project_root()
    model_paths = weights()
    result = {}
    for language in ('English', 'Telugu'):
        original = root / 'reference_source/FastSpeech2'
        if not (original / 'model/fastspeech2.py').exists():
            raise RuntimeError('Packaged reference_source/FastSpeech2 is missing.')
        repo = CACHE / ('repository_' + language)
        if not (repo / '.ready').exists():
            if repo.exists():
                shutil.rmtree(repo)
            shutil.copytree(original, repo)
            for patchdir in [root / 'upstream_modified', root / language / 'upstream_modified']:
                if patchdir.exists():
                    for patch in patchdir.rglob('*.py'):
                        dest = repo / patch.relative_to(patchdir)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(patch, dest)
            (repo / '.ready').write_text(COMMIT)
        result[language] = dict(project=str(root), repository=str(repo),
                                checkpoint=str(model_paths[language]), vocoder=str(model_paths['vocoder']))
    return result
