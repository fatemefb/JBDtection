"""Regressions for native-to-OCR fallback and repeated SPARE occurrences.

Run: PYTHONPATH=apps/backend python -m unittest discover -s apps/backend/tests -p test_pdf_extraction_regressions.py -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pandas as pd

from jb_detection.annotator import CATEGORY_COLORS_RGB, PDFAnnotator
from jb_detection.config import Config
from jb_detection.excel_exporter import ExcelExporter
from jb_detection.models import JBDetectionResult, OcrDetection
from jb_detection.pattern_matcher import PatternMatcher
from jb_detection.tag_matcher import TagMatcher
from jb_detection.unified_pdf_processor import UnifiedPdfProcessor


def detection(text, y=20):
    return OcrDetection(text, 0.99, [], (20, y, 100, 20))


class PdfExtractionRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = Config(pdf_dpi=72)

    def pdf(self, texts):
        path = self.directory / 'input.pdf'
        with fitz.open() as doc:
            for text in texts:
                page = doc.new_page()
                if text:
                    page.insert_text((30, 50), text)
            doc.save(path)
        return str(path)

    def processor(self):
        detector = MagicMock()
        detector.detect.return_value = [detection('SPARE', 40), detection('SPARE', 80)]
        return UnifiedPdfProcessor(config=self.config, detector=detector), detector

    def test_empty_native_extraction_falls_back_to_ocr(self):
        path = self.pdf(['A digital page with a sufficiently long text layer'])
        proc, detector = self.processor()
        with patch.object(proc._digital_extractor, 'extract_from_page', return_value=[]):
            result = proc.process_pdf(path)
        detector.detect.assert_called_once()
        self.assertEqual(len(result[1].spare_identifiers), 2)
        self.assertEqual(proc.pages_scanned, 1)
        self.assertEqual(proc.pages_digital, 0)
        self.assertEqual(proc.total_detections, 2)

    def test_broken_native_text_falls_back_even_with_a_pattern_match(self):
        path = self.pdf(['A digital page with a sufficiently long text layer'])
        proc, detector = self.processor()
        broken = [detection('SPARE ' + '\ufffd' * 30)]
        with patch.object(proc._digital_extractor, 'extract_from_page', return_value=broken):
            proc.process_pdf(path)
        detector.detect.assert_called_once()
        self.assertEqual(proc.total_detections, 2)

    def test_title_only_native_text_falls_back_to_ocr(self):
        proc, detector = self.processor()
        proc.process_pdf(self.pdf(['Electrical drawing general information title']))
        detector.detect.assert_called_once()

    def test_mixed_pdf_selects_strategy_per_page(self):
        proc, detector = self.processor()
        proc.process_pdf(self.pdf(['SPARE\nSPARE\nSPARE\nSPARE\nSPARE\nSPARE', '']))
        detector.detect.assert_called_once()
        self.assertEqual(proc.pages_digital, 1)
        self.assertEqual(proc.pages_scanned, 1)
        self.assertEqual(proc.total_spares, 8)

    def test_ocr_failure_is_counted_as_failed_page(self):
        proc, detector = self.processor()
        detector.detect.side_effect = RuntimeError('GPU test failure')
        with self.assertLogs('jb_detection.unified_pdf_processor', level='ERROR') as logs:
            result = proc.process_pdf(self.pdf(['']))
        self.assertEqual(proc.pages_failed, 1)
        self.assertEqual(proc.pages_scanned, 1)
        self.assertEqual(result[1].spare_identifiers, [])
        self.assertIn('GPU test failure', '\n'.join(logs.output))

    def test_repeated_spares_keep_boxes_numbers_and_legacy_tuple(self):
        matcher = PatternMatcher()
        source = [detection('SPARE', y) for y in (100, 20, 60)]
        result = matcher.match(source)
        self.assertEqual(result.spare_identifiers, ['SPARE'] * 3)
        self.assertEqual(result.tag_to_number, {'SPARE_2': 1, 'SPARE_3': 2, 'SPARE_1': 3})
        self.assertEqual([info.bbox for info in result.tag_match_info.values()], [d.bbox for d in source])
        legacy = JBDetectionResult.from_tuple(result.to_tuple())
        counts = PDFAnnotator(config=self.config).annotate_pdf(
            self.pdf(['']), {1: legacy}, str(self.directory / 'annotated.pdf'),
            tag_to_number={'SPARE_1': 99},
        )
        self.assertEqual(counts['spares'], 3)
        with fitz.open(self.directory / 'annotated.pdf') as doc:
            self.assertEqual(len([d for d in doc[0].get_drawings() if d['color'] is not None]), 3)
            self.assertIn('SPARE #3', doc[0].get_text())
            self.assertNotIn('#99', doc[0].get_text())

    def test_spare_count_in_one_label_or_adjacent_pdf_spans(self):
        matcher = PatternMatcher(spare_examples='SPARE')
        result = matcher.match([
            detection('4 SPARES', 20),
            OcrDetection('3', 0.99, [], (20, 70, 24, 20)),
            OcrDetection('SPARE', 0.99, [], (60, 70, 100, 20)),
            detection('PLUGS', 70),
        ])
        # A count is trusted only when the page has a single SPARE label.
        self.assertEqual(len(result.spare_identifiers), 2)
        self.assertEqual(len(result.spare_positions), 2)
        self.assertEqual(len(result.tag_to_number), 2)

    def test_configured_jb_prefix_keeps_optional_area_digits(self):
        for prefix in ('1201JR', 'JR'):
            matcher = PatternMatcher(jb_examples=prefix)
            result = matcher.match([
                detection('1201JR501'), detection('1201 JR502', 50),
                detection('JR503', 80), detection('1202JR504', 110),
            ])
            expected = ({'1201JR501', '1201JR502', '1201JR503'} if prefix == '1201JR'
                        else {'1201JR501', '1201JR502', 'JR503', '1202JR504'})
            self.assertEqual(result.jb_identifiers, expected)

    def test_short_jb_and_mc_patterns_accept_multisegment_identifiers(self):
        matcher = PatternMatcher(jb_examples='JB', mc_examples='MC')
        result = matcher.match([
            detection('JB-DIA-100-001'),
            detection('MC-DIA-100-001', 50),
            detection('JB To ITR-DCS U-100', 80),
        ])
        self.assertEqual(result.jb_identifiers, {'JB-DIA-100-001'})
        self.assertEqual(result.mc_identifiers, {'MC-DIA-100-001'})

        # A configured prefix followed by unpunctuated prose must not consume
        # later numbers as though the whole phrase were one identifier.
        tag_prefix = PatternMatcher(jb_examples='1201JR')
        self.assertEqual(tag_prefix.match([detection('1201JRS01')]).jb_identifiers, set())

    def test_jb_and_mc_identifiers_allow_attached_terminal_letters(self):
        matcher = PatternMatcher(jb_examples='JSF,JDF', mc_examples='NC')
        result = matcher.match([
            detection('JSF-227S'), detection('JDF-215S', 50),
            detection('NC-JSF-227S', 80),
        ])
        self.assertEqual(result.jb_identifiers, {'JSF-227S', 'JDF-215S'})
        self.assertEqual(result.mc_identifiers, {'NC-JSF-227S'})

    def test_page61_header_and_space_separated_numeric_tag_suffixes(self):
        matcher = PatternMatcher(jb_examples='JAF', mc_examples='NC', spare_examples='SPARE')
        result = matcher.match([
            detection('JAF-225S'), detection('NC-JAF-225S', 40),
            detection('FT-2150', 100), detection('FV-2150', 130),
            detection('TT-2232', 160), detection('FV-2233 1', 190),
            detection('FV-2233 2', 220), detection('TV-7071A', 250),
            detection('TV-7071B', 280), detection('HV-2225 1', 310),
            detection('HV-2225 2', 340), detection('FT-2111', 370),
            detection('PV-2224', 400),
        ])
        self.assertEqual(result.jb_identifiers, {'JAF-225S'})
        self.assertEqual(result.mc_identifiers, {'NC-JAF-225S'})
        self.assertEqual(result.tags, {
            'FT-2150', 'FV-2150', 'TT-2232', 'FV-2233 1', 'FV-2233 2',
            'TV-7071A', 'TV-7071B', 'HV-2225 1', 'HV-2225 2',
            'FT-2111', 'PV-2224',
        })

        # The IO/profile candidate extractor may initially return the base;
        # the space separated digit must remain attached to the OCR identity.
        io_matcher = TagMatcher(config=self.config)
        io_matcher.add_reference_tag('FV-2233')
        matcher.io_tag_matcher = io_matcher
        with_io = matcher.match([detection('FV-2233 1')])
        self.assertEqual(with_io.tags, {'FV-2233 1'})

        ocr_separator_variant = PatternMatcher().match([detection('Fv.2233 1')])
        self.assertEqual(ocr_separator_variant.tags, {'FV-2233 1'})

        output = str(self.directory / 'page61-tags.xlsx')
        ExcelExporter(config=self.config).create_intermediate_excel(
            {'page61.pdf': {61: result.to_tuple()}}, output,
        )
        rows = pd.read_excel(output)
        names = set(rows['Tag/SPARE'].astype(str))
        self.assertTrue({'FV-2233 1', 'FV-2233 2', 'HV-2225 1', 'HV-2225 2'}.issubset(names))

    def test_date_and_ingress_rating_are_not_recovered_as_tags(self):
        matcher = PatternMatcher()
        io_matcher = TagMatcher(config=self.config)
        io_matcher.add_reference_tag('PT-1234')
        matcher.io_tag_matcher = io_matcher

        result = matcher.match([
            detection('02-Jul-2025'), detection('22-May-2023', 50),
            detection('2025-07-02', 80), detection('IP65', 110),
            detection('PT-1234', 140),
        ])

        self.assertEqual(result.tags, {'PT-1234'})
        for token in ('02-JUL-2025', '22-MAY-2023', '2025-07-02', 'IP65', 'IP-65'):
            self.assertTrue(matcher._is_non_tag_pattern(token), token)
        self.assertFalse(matcher._is_non_tag_pattern('PT-1234'))

    def test_numeric_prefixed_jb_example_accepts_split_prefix(self):
        matcher = PatternMatcher(jb_examples='1201JM')
        result = matcher.match([detection('JM506')])
        # The configured numeric area prefix is part of the JB identity even
        # when OCR split or dropped it from this text span.
        self.assertEqual(result.jb_identifiers, {'1201JM506'})
        self.assertEqual(result.tags, set())

    def test_explicitly_empty_patterns_disable_categories(self):
        matcher = PatternMatcher(jb_examples='', mc_examples='', spare_examples='', cable_examples='')
        result = matcher.match([detection('JB-101'), detection('MC-101', 50),
                                detection('SPARE', 80), detection('FRT-1', 110)])
        self.assertEqual(result.jb_identifiers, set())
        self.assertEqual(result.mc_identifiers, set())
        self.assertEqual(result.spare_identifiers, [])

    def test_repeated_identical_tags_share_number_and_spares_each_get_a_number(self):
        result = PatternMatcher().match([
            detection('TE-5223', 10), detection('TE-5223', 30),
            detection('SPARE', 50), detection('SPARE', 70),
        ])
        self.assertEqual(result.tags, {'TE-5223'})
        self.assertEqual(result.tag_to_number, {'TE-5223': 1, 'SPARE_1': 2, 'SPARE_2': 3})
        self.assertEqual(len(result.spare_identifiers), 2)

    def test_custom_spares_export_each_occurrence_with_local_number(self):
        matcher = PatternMatcher(spare_examples=['RESERVE', 'UNUSED'])
        result = matcher.match([detection('RESERVE', 100), detection('UNUSED', 20), detection('RESERVE', 60)])
        output = str(self.directory / 'spares.xlsx')
        exporter = ExcelExporter(config=self.config)
        exporter.set_spare_examples('RESERVE,UNUSED')
        exporter.create_intermediate_excel(
            {'input.pdf': {1: result.to_tuple()}}, output,
            master_tag_numbers={'SPARE_1': 99},
        )
        rows = pd.read_excel(output)
        self.assertEqual(rows['Type'].tolist(), ['SPARE'] * 3)
        self.assertEqual(rows['Tag/SPARE'].tolist(), ['UNUSED', 'RESERVE', 'RESERVE'])
        self.assertEqual(rows['Tag_Number'].tolist(), [1, 2, 3])
        self.assertEqual(rows['Tag_Number_Status'].tolist(), ['Assigned (Position-based)'] * 3)

    def test_labels_are_outside_boxes_and_visible_at_page_edges(self):
        with fitz.open() as doc:
            page = doc.new_page(width=240, height=180)
            boxes = [fitz.Rect(20, 50, 120, 65), fitz.Rect(20, 1, 120, 16), fitz.Rect(200, 150, 239, 165)]
            annotator = PDFAnnotator(config=self.config)
            for index, box in enumerate(boxes):
                annotator._draw_box(page, box, CATEGORY_COLORS_RGB['tag'], label=f'TE-987654A #{index}')
            spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                     for line in block['lines'] for span in line['spans']]
            self.assertEqual(len(spans), 3)
            for span, box in zip(spans, boxes):
                label = fitz.Rect(span['bbox'])
                self.assertFalse(label.intersects(box))
                self.assertTrue(page.rect.contains(label))
                self.assertGreaterEqual(span['size'], 7)
                self.assertEqual(span['color'], 0)
            self.assertLess(spans[0]['bbox'][3], boxes[0].y0)
            self.assertGreater(spans[1]['bbox'][1], boxes[1].y1)

    def test_only_final_cable_is_annotated_with_cable_color(self):
        result = PatternMatcher().match([
            detection('NC-0-1-2-C-3-BL', 30),
            detection('NC-12-3-4-A-5-WHT', 70),
            detection('TE-5223', 110),
        ])
        output = self.directory / 'cables.xlsx'
        ExcelExporter(config=self.config).create_intermediate_excel({'input.pdf': {1: result}}, str(output))
        selected = pd.read_excel(output)['Cable_Code'].iloc[0]
        self.assertEqual(selected, 'NC-12-3-4-A-5-WHT')
        with fitz.open() as doc:
            page = doc.new_page()
            counts = PDFAnnotator(config=self.config)._annotate_page(page, result, result.tag_to_number, 1)
            self.assertEqual(counts['cables'], 1)
            self.assertEqual(counts['tags'], 1)
            self.assertIn(selected, page.get_text())
            self.assertNotIn('NC-0-1-2-C-3-BL', page.get_text())
            cable_boxes = [d for d in page.get_drawings() if d['color'] == CATEGORY_COLORS_RGB['cable']]
            self.assertEqual(len(cable_boxes), 1)
            self.assertEqual(cable_boxes[0]['rect'], fitz.Rect(20, 70, 120, 90))

    def test_configured_cable_unit_is_brand_agnostic_and_mc_local(self):
        matcher = PatternMatcher(cable_examples='12P')
        result = matcher.match([
            OcrDetection('MC-101', 0.99, [], (20, 40, 80, 20)),
            OcrDetection('FRS-12P x 1.5mm2', 0.99, [], (30, 75, 130, 20)),
            OcrDetection('FRT-24P x 1.5mm2', 0.99, [], (500, 75, 130, 20)),
        ])
        self.assertEqual(result.mc_identifiers, {'MC-101'})
        self.assertEqual(result.cable_descriptions, ['12 pair'])
        self.assertEqual(result.raw_cable_descriptions, ['FRS-12P X 1.5MM2'])
        self.assertNotIn('FRS-12P X 1.5MM2', result.tags)
        self.assertNotIn('FRT-24P X 1.5MM2', result.tags)

    def test_configured_cable_unit_joins_adjacent_ocr_spans(self):
        matcher = PatternMatcher(cable_examples='12P')
        result = matcher.match([
            OcrDetection('MC-101', 0.99, [], (20, 40, 80, 20)),
            OcrDetection('12', 0.99, [], (30, 75, 20, 20)),
            OcrDetection('PAIR', 0.99, [], (54, 75, 44, 20)),
        ])
        self.assertEqual(result.cable_descriptions, ['12 pair'])
        self.assertEqual(result.raw_cable_descriptions, ['12 PAIR'])

    def test_configured_cable_without_mc_local_candidate_is_not_exported(self):
        result = PatternMatcher(cable_examples='12P').match([
            OcrDetection('MC-101', 0.99, [], (20, 40, 80, 20)),
            OcrDetection('FRS-12P', 0.99, [], (500, 400, 100, 20)),
        ])
        self.assertEqual(result.cable_descriptions, [])

    def test_separator_variants_share_one_tag_number(self):
        numbers = PatternMatcher().assign_tag_numbers_by_position([
            {'tag': 'TE-5223', 'y': 20, 'x': 20},
            {'tag': 'TE_5223', 'y': 40, 'x': 20},
        ])
        self.assertEqual(numbers, {'TE-5223': 1, 'TE_5223': 1})

    def test_cable_on_same_row_does_not_consume_tag_number(self):
        matcher = PatternMatcher(jb_examples='JB', mc_examples='IC', cable_examples='12P')
        numbers = matcher.assign_tag_numbers_by_position([
            {'tag': 'FRT-12PX2.5MM2', 'y': 20, 'x': 20},
            {'tag': 'FT-12118', 'y': 20, 'x': 500},
            {'tag': 'FT-12218', 'y': 40, 'x': 500},
        ])
        self.assertEqual(numbers, {'FT-12118': 1, 'FT-12218': 2})

    def test_spare_and_tag_share_legacy_position_order(self):
        result = PatternMatcher().match([detection('SPARE', 20), detection('TE-5223', 40)])
        self.assertEqual(result.tag_to_number, {'SPARE_1': 1, 'TE-5223': 2})

    def test_spare_count_is_used_only_for_a_single_spare_label(self):
        matcher = PatternMatcher(spare_examples='SPARE')
        counted = matcher.match([detection('4 SPARE', 20), detection('4 SPARE', 22)])
        self.assertEqual(len(counted.spare_identifiers), 4)
        multiple_labels = matcher.match([detection('4 SPARE', 20), detection('SPARE', 60)])
        self.assertEqual(len(multiple_labels.spare_identifiers), 2)

    def test_spare_count_can_follow_the_single_label(self):
        result = PatternMatcher(spare_examples='SPARE').match([detection('SPARE 4', 20)])
        self.assertEqual(len(result.spare_identifiers), 4)

    def test_sp_fragments_do_not_create_phantom_spare_excel_rows(self):
        result = PatternMatcher().match([
            detection('SPARE', 20), detection('SP', 40), detection('SP 2', 60), detection('SPARE 3', 80),
        ])
        self.assertEqual(result.spare_identifiers, ['SPARE', 'SPARE 3'])
        output = self.directory / 'no-phantom-sp.xlsx'
        ExcelExporter(config=self.config).create_intermediate_excel({'input.pdf': {1: result}}, str(output))
        rows = pd.read_excel(output)
        self.assertEqual(rows['Tag/SPARE'].tolist(), ['SPARE', 'SPARE 3'])
        with fitz.open() as doc:
            counts = PDFAnnotator(config=self.config)._annotate_page(doc.new_page(), result, {}, 1)
            self.assertEqual(counts['spares'], len(rows))
        explicit = PatternMatcher(spare_examples=['SP']).match([detection('SP', 20)])
        self.assertEqual(explicit.spare_identifiers, ['SP'])


if __name__ == '__main__':
    unittest.main()
