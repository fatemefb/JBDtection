from flask import Flask, render_template, request, jsonify
from DataAnalysisModule import TagJBExtractor
import os
import tempfile
import logging

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'frontend', 'static')


# Configure upload folder for temporary files
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Default tesseract path
DEFAULT_TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@app.route('/')
def home():
    return render_template('JB.html')

@app.route('/process', methods=['POST'])
def process_files():
    try:
        # Get PDF and Excel files
        pdf_files = request.files.getlist('pdf_files')
        excel_file = request.files['excel_file']
        output_dir = request.form.get('output_path', '')
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save files temporarily
        pdf_paths = []
        for pdf in pdf_files:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
            pdf.save(temp_path)
            pdf_paths.append(temp_path)
            logger.info(f"Saved PDF: {pdf.filename}")
        
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        excel_file.save(excel_path)
        logger.info(f"Saved Excel file: {excel_file.filename}")
        
        # Create output paths
        output_excel_path = os.path.join(output_dir, 'output.xlsx')
        output_pdf_dir = os.path.join(output_dir, 'annotated_pdfs')
        os.makedirs(output_pdf_dir, exist_ok=True)
        
        # Initialize TagJBExtractor
        logger.info("Initializing TagJBExtractor...")
        extractor = TagJBExtractor(tesseract_path=DEFAULT_TESSERACT_PATH)
        
        # Process files with annotated PDFs
        logger.info("Starting PDF and Excel processing...")
        unmatched_excel_tags, unmatched_pdf_tags = extractor.run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=output_excel_path,
            output_pdf_dir=output_pdf_dir
        )
        
        # Get list of generated annotated PDFs
        annotated_pdfs = [f for f in os.listdir(output_pdf_dir) if f.startswith('annotated_')]
        
        # Clean up temporary files
        for path in pdf_paths:
            os.remove(path)
        os.remove(excel_path)
        
        # Prepare detailed response
        response = {
            'status': 'success',
            'message': 'Processing completed successfully',
            'details': {
                'input_files': {
                    'pdf_count': len(pdf_paths),
                    'pdf_names': [os.path.basename(p) for p in pdf_paths],
                    'excel_file': excel_file.filename
                },
                'output_files': {
                    'excel_path': output_excel_path,
                    'annotated_pdfs_dir': output_pdf_dir,
                    'annotated_pdfs': annotated_pdfs
                },
                'results': {
                    'unmatched_excel_tags': unmatched_excel_tags,
                    'unmatched_pdf_tags': unmatched_pdf_tags,
                    'unmatched_excel_count': len(unmatched_excel_tags),
                    'unmatched_pdf_count': len(unmatched_pdf_tags)
                }
            }
        }
        
        logger.info("Processing completed successfully")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': {
                'error_type': type(e).__name__,
                'error_description': str(e)
            }
        }), 500

if __name__ == '__main__':
    # Print startup message
    print("=" * 50)
    print("JB Detection System")
    print("=" * 50)
    print(f"Tesseract Path: {DEFAULT_TESSERACT_PATH}")
    print("Starting server...")
    print("=" * 50)
    
    # Run the Flask app
    app.run(debug=True)
