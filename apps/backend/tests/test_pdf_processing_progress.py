"""Progress must reflect completed PDF work without making it fail."""
import tempfile
import unittest
from pathlib import Path

import fitz

from jb_detection.annotator import PDFAnnotator
from jb_detection.config import Config
from jb_detection.models import JBDetectionResult, TagMatchInfo
from jb_detection.unified_pdf_processor import UnifiedPdfProcessor


class PdfProcessingProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'input.pdf'
        with fitz.open() as doc:
            for _ in range(3):
                page = doc.new_page()
                page.insert_text((30, 50), 'SPARE\nSPARE\nSPARE\nSPARE\nSPARE\nSPARE')
            doc.save(self.path)

    def test_extraction_reports_each_completed_page(self):
        events = []
        proc = UnifiedPdfProcessor(config=Config(pdf_dpi=72), progress_callback=events.append)
        results = proc.process_pdf(str(self.path))
        self.assertEqual(set(results), {1, 2, 3})
        self.assertEqual([(event['stage'], event['current'], event['total']) for event in events],
                         [('extract', 0, 3), ('extract', 1, 3), ('extract', 2, 3), ('extract', 3, 3)])
        self.assertTrue(all(event['pdf_path'] == str(self.path) for event in events))
        self.assertEqual(proc.pages_failed, 0)

    def test_annotation_and_save_events_follow_actual_output(self):
        events = []
        results = {page: JBDetectionResult(spare_identifiers=['SPARE'],
            tag_to_number={'SPARE_1': 1}, tag_match_info={'SPARE_1': TagMatchInfo(
                match_type='SPARE', matched_tag='SPARE', bbox=(20, 30, 90, 20))}) for page in range(1, 4)}
        output = self.root / 'annotated.pdf'
        def callback(event):
            events.append(event)
            if event['stage'] == 'save' and event['current'] == 1:
                with fitz.open(output) as doc:
                    self.assertEqual(len(doc), 3)
                    self.assertEqual(len(doc[0].get_drawings()), 1)
        counts = PDFAnnotator(config=Config(pdf_dpi=72), progress_callback=callback).annotate_pdf(
            str(self.path), results, str(output))
        self.assertEqual(counts['spares'], 3)
        self.assertEqual([(event['stage'], event['current'], event['total']) for event in events],
                         [('annotate', 0, 3), ('annotate', 1, 3), ('annotate', 2, 3), ('annotate', 3, 3),
                          ('save', 0, 1), ('save', 1, 1)])
        with fitz.open(output) as doc:
            self.assertIn('SPARE #1', doc[0].get_text())

    def test_failed_status_updates_do_not_lose_results(self):
        def broken_callback(event):
            raise RuntimeError('status store unavailable')
        proc = UnifiedPdfProcessor(config=Config(pdf_dpi=72), progress_callback=broken_callback)
        with self.assertLogs('jb_detection.progress', level='WARNING'):
            results = proc.process_pdf(str(self.path))
        self.assertEqual(proc.pages_failed, 0)
        self.assertEqual(sum(len(result.spare_identifiers) for result in results.values()), 18)


if __name__ == '__main__':
    unittest.main()
