
"""Conservative Telugu normalization and eSpeak pronunciation candidates."""
import re
import subprocess
import unicodedata

def normalize_training_text(text):
    text = unicodedata.normalize("NFC", text)
    unsupported = sorted({
        c for c in text
        if not (
            "\u0c00" <= c <= "\u0c7f"
            or c.isspace()
            or unicodedata.category(c).startswith("P")
        )
    })
    if unsupported:
        raise ValueError(
            "Characters requiring review: "
            + ", ".join(f"U+{ord(c):04X}" for c in unsupported)
        )

    text = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Empty normalized text")
    return text

def phonemize_word(word):
    result = subprocess.run(
        ["espeak-ng", "-q", "-v", "te", "--ipa", "--sep=|", word],
        capture_output=True, text=True, check=True,
    )
    raw = result.stdout.strip()
    if not raw:
        raise ValueError(f"Empty pronunciation for {word!r}")

    tokens = [
        token.replace("ˈ", "").replace("ˌ", "")
        for token in re.split(r"[|\s]+", raw) if token
    ]
    tokens = [token for token in tokens if token]

    if not tokens or any(
        "\u200d" in token or "\u200c" in token for token in tokens
    ):
        raise ValueError(f"Unexpected pronunciation encoding: {raw!r}")
    return tokens
