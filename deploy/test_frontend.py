import tempfile
from pathlib import Path
import unittest
from deploy.frontend import validate_text, english_phones, telugu_phones, encode_phones, load_lexicon
from deploy.bootstrap import sha256


class FrontendTests(unittest.TestCase):
    def test_blank(self):
        with self.assertRaises(ValueError): validate_text('   ', 'English')

    def test_language(self):
        with self.assertRaises(ValueError): validate_text('Hello', 'Other')

    def test_length(self):
        with self.assertRaises(ValueError): validate_text('a' * 251, 'English')

    def test_numbers(self):
        with self.assertRaises(ValueError): validate_text('It is 10:45.', 'English')

    def test_unicode_numbers(self):
        with self.assertRaises(ValueError): validate_text('ఇది ౨.', 'Telugu')

    def test_telugu_language(self):
        with self.assertRaises(ValueError): validate_text('Hello', 'Telugu')

    def test_english_language(self):
        with self.assertRaises(ValueError): validate_text('తెలుగు', 'English')

    def test_controls(self):
        with self.assertRaises(ValueError): validate_text('Hi\x00there', 'English')

    def test_punctuation_only(self):
        with self.assertRaises(ValueError): validate_text('!!!', 'English')

    def test_basic_valid(self):
        self.assertEqual(validate_text(' Hello. ', 'English'), 'Hello.')
        self.assertEqual(validate_text('తెలుగు.', 'Telugu'), 'తెలుగు.')

    def test_english_dictionary_and_fallback(self):
        result = english_phones('Hi, cat!', {'hi': ['HH', 'AY1']}, lambda w: ['K', 'AE1', 'T'])
        self.assertEqual(result, ['HH', 'AY1', 'sp', 'K', 'AE1', 'T'])

    def test_telugu_mapping(self):
        result = telugu_phones('తెలుగు.', {'t': 'TEBB', 'e': 'TEAK'}, lambda w: ['t', 'e'])
        self.assertEqual(result, ['TEBB', 'TEAK'])

    def test_unknown_telugu(self):
        with self.assertRaises(ValueError): telugu_phones('తెలుగు', {}, lambda w: ['ŋ'])

    def test_exact_encoding(self):
        self.assertEqual(encode_phones(['HH', 'AY1'], ['_', '@AY1', '@HH']), [2, 1])
        with self.assertRaises(ValueError): encode_phones(['NOTAPHONE'], ['_'])
        with self.assertRaises(ValueError): encode_phones([], ['_'])

    def test_lexicon_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'lexicon.dict'
            p.write_text('hi 0.5 HH AY1\nhi HH IH1\ncat K AE1 T\n')
            result = load_lexicon(p)
            self.assertEqual(result['hi'], ['HH', 'AY1'])
            self.assertEqual(result['cat'], ['K', 'AE1', 'T'])

    def test_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'a'
            p.write_bytes(b'abc')
            self.assertEqual(sha256(p), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')


if __name__ == '__main__':
    unittest.main()
