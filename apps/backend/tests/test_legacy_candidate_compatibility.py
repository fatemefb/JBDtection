"""Check the migrated scoring against the legacy source without loading OCR."""
import ast
from pathlib import Path
import re
import types
import unittest
from typing import Any, Dict, List, Set

from jb_detection.legacy_candidates import LegacyCandidateRules
from jb_detection.tag_matcher import TagMatcher, _lev


class LegacyCandidateCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).parents[1] / 'DataAnalysisModule.py').read_text()
        tree = ast.parse(source)
        names = {'_normalize_ocr_tag_candidate', '_build_io_pattern_profile',
                 '_score_pattern_candidate', '_extract_tag_prefix'}
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {'re': re, 'Levenshtein': _lev, 'Any': Any, 'Dict': Dict, 'List': List, 'Set': Set}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<legacy-candidate-functions>', 'exec'), namespace)
        cls.legacy = types.SimpleNamespace(_is_non_tag_pattern=lambda tag: False)
        for name in names:
            setattr(cls.legacy, name, types.MethodType(namespace[name], cls.legacy))
        # The prefix-learning block is separate from the legacy score helper.
        start = source.index('        _io_prefixes_set = set()')
        stop = source.index('        logger.info(', start)
        import textwrap
        cls.prefix_learning = compile(textwrap.dedent(source[start:stop]), '<legacy-prefix-learning>', 'exec')

    def test_profile_scores_and_acceptance_match_legacy(self):
        fixtures = [
            ({'TE-5223', '11-FV-301', '21HS-001'},
             ['TE-5224', 'PT-5223', '11-XV-301', 'TE-5224ABC', 'TE-987654A', '21ZZ-001', 'XYZ-12345678-ABC']),
            ({'LUSY-2474A', 'USY-2482A'},
             ['LUSY-99999B', 'USY-1', 'USY-876543C', 'XYZ-2474A', 'UY-5221']),
            ({'11SAM10AN020XB91', '11HOTSPARE-DO22-03'},
             ['12SAM10AN030XB92', '11SAM10AN020XB91', '11HOTSPARE-DO23-04', '12SPARE-DO21-15']),
            ({'HY-11306-SA', 'LY-21201-B-SB', 'PT-21205'},
             ['HY-11307-SB', 'LY-99999-A-SA', 'LT-21206-E', 'HY11306SA']),
        ]
        general = re.compile(r'^[A-Z0-9]{2,6}[-]?[A-Z0-9]{1,10}(?:[-][A-Z0-9]{1,10}){0,3}$')
        for references, tags in fixtures:
            rules = LegacyCandidateRules(references, _lev.distance)
            profile = self.legacy._build_io_pattern_profile(references)
            self.assertEqual(rules.profile, profile)
            namespace = {'re': re, 'io_list_tags': references}
            exec(self.prefix_learning, namespace)
            for tag in tags:
                with self.subTest(references=references, tag=tag):
                    score = self.legacy._score_pattern_candidate(tag, profile)
                    self.assertEqual(rules._score_pattern_candidate(tag, rules.profile), score)
                    normalized = self.legacy._normalize_ocr_tag_candidate(tag)
                    expected = bool(general.fullmatch(normalized) or score >= 0.62) and (
                        score >= 0.62 or any(pattern.match(normalized) for pattern in namespace['_io_regex_patterns']))
                    self.assertEqual(rules.accepts(tag), expected)

    def test_structural_candidates_require_a_learned_family(self):
        matcher = TagMatcher()
        for tag in ['TE-5223', '11-FV-301', '21HS-001']:
            matcher.add_reference_tag(tag)
        for tag in ['PT-5223', '11-XV-301', 'TE-5224ABC']:
            self.assertEqual(matcher.match_tag(tag)[0], 'unmatched')
            self.assertEqual(matcher.extract_candidates(tag), [])
        self.assertEqual(matcher.match_tag('TE-5224')[0], 'similar')
        self.assertFalse(matcher.matches_io_pattern('text TE-5224'))


if __name__ == '__main__':
    unittest.main()
