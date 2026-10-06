"""Regressions for yellow highlights and table rules crossing OCR text."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import fitz
import numpy as np

from jb_detection.config import Config, load_config
from jb_detection.image_preprocessor import _normalize_highlights, _remove_ruling_lines, preprocess
from jb_detection.models import OcrDetection
from jb_detection.ocr_recovery import detect_with_jb_recovery
from jb_detection.pattern_matcher import PatternMatcher
from jb_detection.unified_pdf_processor import UnifiedPdfProcessor


class ImagePreprocessingTests(unittest.TestCase):
    def test_yellow_background_is_whitened_but_ink_is_preserved(self):
        image = np.full((120, 320, 3), 255, np.uint8)
        image[20:100, 20:300] = (0, 255, 255)
        image[40:70, 50:55] = (0, 0, 0)
        image[40:70, 80:85] = (0, 50, 50)  # dark stroke under highlighter
        image[40:70, 110:115] = (0, 0, 255)  # colored ink
        original = image.copy()
        result = _normalize_highlights(image)
        np.testing.assert_array_equal(result[25, 25], [255, 255, 255])
        for x in [50, 80, 110]:
            np.testing.assert_array_equal(result[50, x], original[50, x])
        np.testing.assert_array_equal(image, original)

    def test_horizontal_and_vertical_rules_are_removed_without_erasing_crossing_strokes(self):
        gray = np.full((180, 340), 255, np.uint8)
        cv2.line(gray, (10, 70), (325, 70), 0, 2)
        cv2.line(gray, (170, 10), (170, 165), 0, 2)
        cv2.line(gray, (80, 45), (80, 100), 0, 2)  # vertical glyph stroke
        cv2.line(gray, (145, 115), (195, 115), 0, 2)  # horizontal glyph stroke
        result = _remove_ruling_lines(gray)
        self.assertGreater(result[70, 30], 200)
        self.assertGreater(result[30, 170], 200)
        self.assertEqual(result[70, 80], 0)
        self.assertEqual(result[115, 170], 0)
        self.assertEqual(result.shape, gray.shape)
        self.assertEqual(gray[70, 30], 0)

    def test_short_glyph_bars_are_unchanged(self):
        gray = np.full((140, 400), 255, np.uint8)
        cv2.putText(gray, 'JB-101', (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 2, cv2.LINE_AA)
        np.testing.assert_array_equal(_remove_ruling_lines(gray), gray)

    def test_broad_dark_objects_are_not_treated_as_rules(self):
        gray = np.full((180, 340), 255, np.uint8)
        gray[40:80, 20:310] = 0
        np.testing.assert_array_equal(_remove_ruling_lines(gray), gray)

    def test_cleaning_preserves_output_coordinates_and_grayscale_input(self):
        for image in [np.full((140, 400), 255, np.uint8), np.full((140, 400, 3), 255, np.uint8)]:
            result = preprocess(image)
            self.assertEqual(result.shape, (140, 400, 3))
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result == 255))

    def test_cleaning_can_be_disabled_in_configuration(self):
        with patch.dict(os.environ, {'JBDET_PREPROCESS_REMOVE_HIGHLIGHTS': 'false',
                                    'JBDET_PREPROCESS_REMOVE_RULING_LINES': 'false'}):
            config = load_config()
        self.assertFalse(config.preprocess_remove_highlights)
        self.assertFalse(config.preprocess_remove_ruling_lines)
        with patch('jb_detection.image_preprocessor._normalize_highlights') as highlights, \
                patch('jb_detection.image_preprocessor._remove_ruling_lines') as lines:
            preprocess(np.full((100, 200, 3), 255, np.uint8), config=config)
        highlights.assert_not_called()
        lines.assert_not_called()

    def test_scanned_pipeline_sends_cleaned_image_to_ocr_and_keeps_jb_box(self):
        source = np.full((140, 400, 3), 255, np.uint8)
        source[20:110, 10:390] = (0, 255, 255)
        cv2.putText(source, 'JB-101', (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(source, (5, 65), (395, 65), (0, 0, 0), 2)
        detector = MagicMock()
        def detect(image):
            self.assertEqual(image.shape, source.shape)
            self.assertTrue(np.all(image[25, 15] == 255))
            self.assertTrue(np.all(image[65, 15] == 255))
            return [OcrDetection('JB-101', .95, [], (80, 55, 150, 32))]
        detector.detect.side_effect = detect
        processor = UnifiedPdfProcessor(config=Config(pdf_dpi=72, preprocess_remove_ruling_lines=True), detector=detector)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.pdf'
            with fitz.open() as document:
                document.new_page()
                document.save(path)
            with patch('jb_detection.image_preprocessor.load_pdf_page', return_value=source):
                result = processor.process_pdf(str(path))[1]
        self.assertEqual(result.jb_identifiers, {'JB-101'})
        self.assertEqual(result.tag_match_info['JB-101'].bbox, (80, 55, 150, 32))
        detector.detect.assert_called_once()
        self.assertEqual(processor.pages_failed, 0)

    def recovery_image(self):
        image = np.full((140, 400, 3), 255, np.uint8)
        cv2.putText(image, 'JB-101', (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(image, (5, 65), (395, 65), (0, 0, 0), 2)
        return image

    def test_recovery_adds_only_valid_jb_and_preserves_primary_detections(self):
        image = self.recovery_image()
        original = image.copy()
        primary = OcrDetection('TE-5223', .97, [], (40, 100, 80, 20))
        jb = OcrDetection('JB-101', .9, [], (80, 55, 150, 32))
        detector = MagicMock()
        detector.detect.side_effect = [[primary], [jb, OcrDetection('TE-9999', .99, [], (40, 100, 80, 20))]]
        result = detect_with_jb_recovery(image, detector, PatternMatcher(), Config())
        self.assertEqual(result, [primary, jb])
        self.assertIs(result[0], primary)
        self.assertEqual(detector.detect.call_count, 2)
        np.testing.assert_array_equal(image, original)

    def test_existing_jb_does_not_trigger_recovery(self):
        jb = OcrDetection('JSF-101', .95, [], (80, 55, 150, 32))
        detector = MagicMock()
        detector.detect.return_value = [jb]
        result = detect_with_jb_recovery(self.recovery_image(), detector, PatternMatcher(jb_examples='JSF'), Config())
        self.assertEqual(result, [jb])
        detector.detect.assert_called_once()

    def test_unchanged_view_does_not_repeat_ocr(self):
        detector = MagicMock()
        detector.detect.return_value = []
        detect_with_jb_recovery(np.full((100, 200, 3), 255, np.uint8), detector, PatternMatcher(), Config())
        detector.detect.assert_called_once()

    def test_failed_recovery_keeps_initial_result(self):
        initial = [OcrDetection('TE-5223', .97, [], (40, 100, 80, 20))]
        detector = MagicMock()
        detector.detect.side_effect = [initial, RuntimeError('recovery OCR failure')]
        with self.assertLogs('jb_detection.ocr_recovery', level='WARNING'):
            result = detect_with_jb_recovery(self.recovery_image(), detector, PatternMatcher(), Config())
        self.assertEqual(result, initial)

    def test_recovery_can_be_disabled(self):
        detector = MagicMock()
        detector.detect.return_value = []
        detect_with_jb_recovery(self.recovery_image(), detector, PatternMatcher(), Config(ocr_recover_missing_jb=False))
        detector.detect.assert_called_once()


if __name__ == '__main__':
    unittest.main()
