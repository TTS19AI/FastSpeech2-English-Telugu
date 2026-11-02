""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)


# Assignment: Telugu phone aliases; preserve existing token IDs.
symbols += ['@TEAA', '@TEAB', '@TEAC', '@TEAD', '@TEAE', '@TEAF', '@TEAG', '@TEAH', '@TEAI', '@TEAJ', '@TEAK', '@TEAL', '@TEAM', '@TEAN', '@TEAO', '@TEAP', '@TEAQ', '@TEAR', '@TEAS', '@TEAT', '@TEAU', '@TEAV', '@TEAW', '@TEAX', '@TEAY', '@TEAZ', '@TEBA', '@TEBB', '@TEBC', '@TEBD', '@TEBE', '@TEBF', '@TEBG', '@TEBH', '@TEBI', '@TEBJ', '@TEBK', '@TEBL', '@TEBM', '@TEBN', '@TEBO', '@TEBP', '@TEBQ', '@TEBR', '@TEBS']
