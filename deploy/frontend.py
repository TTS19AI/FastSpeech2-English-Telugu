"""Conservative live-text frontend. No unknown phone is silently dropped."""
import re
import subprocess
import unicodedata

MAX_CHARS = 250
MAX_PHONES = 220


def validate_text(text, language):
    if language not in ('English', 'Telugu'):
        raise ValueError('Select English or Telugu.')
    text = unicodedata.normalize('NFC', str(text or '')).strip()
    if not text:
        raise ValueError('Please enter a sentence.')
    if len(text) > MAX_CHARS:
        raise ValueError(f'Use at most {MAX_CHARS} characters per request.')
    if any(c.isdigit() for c in text):
        raise ValueError('Please write numbers, dates and times as spoken words.')
    if any(unicodedata.category(c).startswith('C') and c not in '\n\r\t' for c in text):
        raise ValueError('Remove control or invisible formatting characters.')
    if language == 'Telugu':
        if any(not ('\u0c00' <= c <= '\u0c7f' or c.isspace() or unicodedata.category(c).startswith('P')) for c in text):
            raise ValueError('For Telugu, use Telugu script and punctuation only.')
    elif any(not (c.isascii() and (c.isalpha() or c.isspace() or unicodedata.category(c).startswith('P'))) for c in text):
        raise ValueError('For English, use English letters and punctuation only.')
    if not any(c.isalpha() for c in text):
        raise ValueError('Please enter words, not only punctuation.')
    return text


def load_lexicon(path):
    lexicon = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        # Support both plain and probability-bearing pronunciation dictionaries.
        phones = parts[1:]
        try:
            float(phones[0])
            phones = phones[1:]
        except ValueError:
            pass
        if phones:
            lexicon.setdefault(parts[0].lower(), phones)
    return lexicon


def english_phones(text, lexicon, g2p):
    phones = []
    for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^\w\s]", text):
        if not token[0].isalpha():
            if phones and phones[-1] != 'sp':
                phones.append('sp')
        else:
            candidates = lexicon.get(token.lower())
            phones.extend(candidates if candidates else [p for p in g2p(token) if p.strip()])
    # Match the original frontend's treatment of trailing punctuation.
    while phones and phones[-1] == 'sp':
        phones.pop()
    return phones


def espeak_word(word):
    result = subprocess.run(['espeak-ng', '-q', '-v', 'te', '--ipa', '--sep=|', word],
                            capture_output=True, text=True, check=True, timeout=15)
    return [p.replace('ˈ', '').replace('ˌ', '') for p in re.split(r'[|\s]+', result.stdout.strip()) if p.replace('ˈ', '').replace('ˌ', '')]


def telugu_phones(text, mapping, phonemizer=espeak_word):
    clean = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    phones = []
    for word in clean.split():
        ipa = phonemizer(word)
        if not ipa:
            raise ValueError('Empty pronunciation returned for a Telugu word.')
        unknown = sorted(set(ipa) - set(mapping))
        if unknown:
            raise ValueError(f'Pronunciation contains untrained phones: {unknown}. Try different wording.')
        phones.extend(mapping[p] for p in ipa)
    return phones


def encode_phones(phones, symbols):
    if not phones or len(phones) > MAX_PHONES:
        raise ValueError(f'Use a shorter sentence (maximum {MAX_PHONES} pronunciation tokens).')
    indices = {s: i for i, s in enumerate(symbols)}
    missing = sorted({p for p in phones if '@' + p not in indices})
    if missing:
        raise ValueError(f'Unsupported pronunciation phones: {missing}')
    return [indices['@' + p] for p in phones]
