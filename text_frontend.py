
"""English synthesis frontend adapted from upstream synthesize.py."""
import re
from string import punctuation

from g2p_en import G2p
from synthesize import read_lexicon
from text import text_to_sequence
from text.symbols import symbols


class EnglishFrontend:
    def __init__(self, lexicon_path):
        self.lexicon = read_lexicon(str(lexicon_path))
        self.g2p = G2p()
        self.symbol_ids = {
            symbol: index for index, symbol in enumerate(symbols)
        }

    def prepare(self, entry, cleaners):
        text = entry["synthesis_text"].rstrip(punctuation)
        overrides = {}
        for override in entry.get("pronunciation_overrides", []):
            word = override["word"].lower()
            if word in overrides:
                raise ValueError(f"Duplicate pronunciation override: {word}")
            overrides[word] = list(override["phones"])

        phones = []
        applied = set()
        for word in re.split(r"([,;.\-\?\!\s+])", text):
            key = word.lower()
            if key in overrides:
                phones.extend(overrides[key])
                applied.add(key)
            elif key in self.lexicon:
                phones.extend(self.lexicon[key])
            else:
                phones.extend(p for p in self.g2p(word) if p != " ")

        if applied != set(overrides):
            raise ValueError("One or more pronunciation overrides were not used.")

        # Preserve the upstream punctuation-to-silence convention.
        phone_text = "{" + "}{".join(phones) + "}"
        phone_text = re.sub(r"\{[^\w\s]?\}", "{sp}", phone_text)
        phone_text = phone_text.replace("}{", " ")
        final_phones = phone_text[1:-1].split()

        if not final_phones:
            raise ValueError("Empty phoneme sequence.")

        unknown = sorted({
            phone for phone in final_phones
            if "@" + phone not in self.symbol_ids
        })
        if unknown:
            raise ValueError(f"Unsupported phones: {unknown}")

        token_ids = text_to_sequence(phone_text, cleaners)
        expected_ids = [
            self.symbol_ids["@" + phone] for phone in final_phones
        ]
        if token_ids != expected_ids:
            raise ValueError("Encoder changed or discarded phoneme tokens.")

        return {
            "id": entry["id"],
            "original_text": entry["original_text"],
            "synthesis_text": entry["synthesis_text"],
            "normalization_rules": entry.get("normalization_rules", []),
            "pronunciation_overrides": entry.get("pronunciation_overrides", []),
            "phones": final_phones,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "tokenization_verified": True,
        }
