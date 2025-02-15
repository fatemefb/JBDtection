import torch
import re 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import pytesseract
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings
import io
from IPython.display import display, Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# BasicConv class for basic convolution operations
class BasicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# DBNet Model Class
class DBNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(DBNet, self).__init__()
        
        # Backbone (ResNet50)
        backbone = models.resnet50(pretrained=pretrained)
        self.stage1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage2 = backbone.layer1
        self.stage3 = backbone.layer2
        self.stage4 = backbone.layer3
        self.stage5 = backbone.layer4
        
        # Neck (FPN)
        self.lateral3 = nn.Conv2d(1024, 256, 1)
        self.lateral4 = nn.Conv2d(2048, 256, 1)
        self.lateral2 = nn.Conv2d(512, 256, 1)
        
        # Head
        self.binarize = nn.Sequential(
            BasicConv(256, 64, 3, padding=1),
            BasicConv(64, 64, 3, padding=1),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.threshold = nn.Sequential(
            BasicConv(256, 64, 3, padding=1),
            BasicConv(64, 64, 3, padding=1),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        
        # FPN
        p5 = self.lateral4(c5)
        p4 = self._upsample_add(p5, self.lateral3(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))
        
        # Head
        binary_map = self.binarize(p3)
        threshold_map = self.threshold(p3)
        
        return binary_map, threshold_map

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=False) + y
    
class JBImageProcessor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Configure tesseract for better recognition
        self.tesseract_config = r'--oem 3 --psm 11'
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Enhanced image preprocessing with better text detection"""
        # Store original image
        original = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal and enhancement
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Edge enhancement
        edges = cv2.Canny(morph, 50, 150)
        enhanced = cv2.addWeighted(morph, 0.7, edges, 0.3, 0)
        
        # Convert to RGB and normalize
        image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_tensor, original

    def analyze_tag_columns(self, image: np.ndarray, initial_tag: Tuple[str, Tuple[int, int, int, int]], 
                          column_tolerance: int = 50) -> List[Dict[str, Union[str, Tuple[int, int, int, int]]]]:
        """Analyze vertical columns for tag detection"""
        tag_text, (x, y, w, h) = initial_tag
        column_tags = []
        
        # Define column boundaries
        column_center = x + w//2
        column_left = max(0, column_center - column_tolerance)
        column_right = min(image.shape[1], column_center + column_tolerance)
        
        # Get the vertical section of the image
        column_section = image[:, column_left:column_right]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(column_section, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours in the column
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding box
            x_rel, y_rel, w_rel, h_rel = cv2.boundingRect(contour)
            
            # Convert coordinates back to original image space
            x_abs = x_rel + column_left
            
            # Extract potential tag region
            roi = image[y_rel:y_rel+h_rel, x_abs:x_abs+w_rel]
            
            # Perform OCR
            text = pytesseract.image_to_string(roi, config=self.tesseract_config)
            text = text.strip()
            
            # Check if text matches tag pattern
            if self.is_valid_tag(text):
                tag_info = {
                    'text': text,
                    'bbox': (x_abs, y_rel, x_abs + w_rel, y_rel + h_rel),
                    'column_center': column_center
                }
                column_tags.append(tag_info)
        
        # Sort tags by vertical position
        column_tags.sort(key=lambda x: x['bbox'][1])
        return column_tags

    def analyze_right_section(self, image: np.ndarray, threshold_ratio: float = 0.6) -> List[Dict[str, Union[str, Tuple[int, int, int, int]]]]:
        """Analyze the right section of the image for JB detection"""
        height, width = image.shape[:2]
        right_start = int(width * threshold_ratio)
        right_section = image[:, right_start:]
        
        # Enhance right section
        gray = cv2.cvtColor(right_section, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        jb_detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_adjusted = x + right_start
            
            # Extract potential JB region
            roi = image[y:y+h, x_adjusted:x_adjusted+w]
            
            # Perform OCR
            text = pytesseract.image_to_string(roi, config=self.tesseract_config)
            text = text.strip()
            
            if self.is_valid_jb(text):
                jb_info = {
                    'text': text,
                    'bbox': (x_adjusted, y, x_adjusted + w, y + h),
                    'confidence': 100
                }
                jb_detections.append(jb_info)
        
        return jb_detections

    def is_valid_tag(self, text: str) -> bool:
        """Check if text matches tag pattern with more flexible matching"""
        tag_patterns = [
            re.compile(r'[A-Z]{2}[-_]?\d{4}[A-Z]?'),  # More flexible pattern
            re.compile(r'[A-Z]{2}\s*[-_]?\s*\d{4}[A-Z]?'),  # Allow spaces
            re.compile(r'[A-Z]{2}[-_.]?\d{4}[A-Z]?'),  # Allow dots
            re.compile(r'[A-Z]{2}\d{4}[A-Z]?')  # No separator
        ]
        text = text.strip().upper()  # Convert to uppercase
        return any(pattern.match(text) for pattern in tag_patterns)

    def is_valid_jb(self, text: str) -> bool:
        """Check if text matches JB pattern with more flexible matching"""
        jb_patterns = [
            re.compile(r'JB[A-Z]{3}[-_]?\d{3}'),  # Standard pattern
            re.compile(r'JB\s*[A-Z]{3}\s*[-_]?\s*\d{3}'),  # Allow spaces
            re.compile(r'JB[A-Z]{3}[-_.]?\d{3}'),  # Allow dots
            re.compile(r'JB[A-Z]{3}\d{3}'),  # No separator
            re.compile(r'J\s*B[A-Z]{3}[-_]?\d{3}')  # Space between J and B
        ]
        text = text.strip().upper()  # Convert to uppercase
        return any(pattern.match(text) for pattern in jb_patterns)

    def detect_tags(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """Enhanced tag detection with column analysis"""
        data = pytesseract.image_to_data(image, config=self.tesseract_config, 
                                        output_type=pytesseract.Output.DICT)
        
        initial_tags = []
        processed_columns = set()
        all_tags = []
        
        # First pass: find initial tags
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            
            if text and conf > 60:  # confidence threshold
                bbox = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i]
                )
                
                if self.is_valid_tag(text):
                    initial_tags.append((text, bbox))
        
        # Second pass: analyze columns for each initial tag
        for initial_tag in initial_tags:
            column_center = initial_tag[1][0] + initial_tag[1][2]//2
            
            if any(abs(col - column_center) < 50 for col in processed_columns):
                continue
                
            column_tags = self.analyze_tag_columns(image, initial_tag)
            all_tags.extend([(tag['text'], tag['bbox']) for tag in column_tags])
            processed_columns.add(column_center)
        
        # Remove duplicates and sort
        unique_tags = list(set(all_tags))
        unique_tags.sort(key=lambda x: (x[1][0], x[1][1]))
        
        return unique_tags
# Continue JBImageProcessor class
    def extract_text_with_coords(self, image_path: str) -> Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]]:
        """Extract text and coordinates with improved tag and JB detection"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Process image
        preprocessed, original_image = self.preprocess_image(image)
        with torch.no_grad():
            binary_map, _ = self.model(preprocessed)
        
        binary_map = binary_map.squeeze().numpy()
        mask = (binary_map > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        masked_image = cv2.bitwise_and(original_image, original_image, mask=mask_resized)
        
        # Detect tags in columns
        tag_results = self.detect_tags(masked_image)
        
        # Detect JBs in right section
        jb_detections = self.analyze_right_section(original_image)
        jbs = [(jb['text'], jb['bbox']) for jb in jb_detections]
        
        return {
            "tags": tag_results,
            "jbs": jbs,
            "original_image": original_image,
            "masked_image": masked_image,
            "tag_columns": self.get_tag_columns(tag_results)
        }

    def get_tag_columns(self, tags: List[Tuple[str, Tuple[int, int, int, int]]]) -> Dict[int, List[str]]:
        """Group tags by columns"""
        columns = {}
        
        for tag, bbox in tags:
            column_center = bbox[0] + (bbox[2] - bbox[0])//2
            
            # Find or create column
            column_key = None
            for existing_center in columns.keys():
                if abs(existing_center - column_center) < 50:  # 50px tolerance
                    column_key = existing_center
                    break
            
            if column_key is None:
                column_key = column_center
                columns[column_key] = []
                
            columns[column_key].append(tag)
        
        # Sort tags in each column
        for column in columns.values():
            column.sort()
            
        return columns

    def create_output_image(self, results: Dict) -> np.ndarray:
        """Create visualization of detection results"""
        if "original_image" not in results:
            raise ValueError("Original image not found in results")
            
        display_image = results["original_image"].copy()
        
        # Draw boxes for tags (green)
        for tag, bbox in results.get('tags', []):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_image, tag, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw boxes for JBs (blue)
        for jb, bbox in results.get('jbs', []):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_image, jb, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # Draw column lines if available
        if 'tag_columns' in results:
            for column_center in results['tag_columns'].keys():
                cv2.line(display_image, 
                        (column_center, 0),
                        (column_center, display_image.shape[0]),
                        (0, 255, 255), 1)  # Yellow line
                
        return display_image

class DataAnalysisModule:
    def __init__(self, excel_path: str):
        """Initialize with Excel file containing tag data"""
        self.data = pd.read_excel(excel_path)
        self.jb_tag_mapping: Dict[str, List[str]] = {}

    def analyze_tags(self) -> None:
        """Analyze tags and create JB to tag mapping"""
        for _, row in self.data.iterrows():
            jb_name = str(row.get('JB Name', ''))
            tag_no = str(row.get('Tag No', ''))
            
            if jb_name and tag_no:
                if jb_name not in self.jb_tag_mapping:
                    self.jb_tag_mapping[jb_name] = []
                self.jb_tag_mapping[jb_name].append(tag_no)

    def validate_detected_text(self, detected_data: Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]]) -> Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]]:
        """Validate detected JBs and Tags against Excel data"""
        valid_tags = [(tag, bbox) for tag, bbox in detected_data.get("tags", []) 
                     if any(tag in tags for tags in self.jb_tag_mapping.values())]
        valid_jbs = [(jb, bbox) for jb, bbox in detected_data.get("jbs", []) 
                    if jb in self.jb_tag_mapping]

        return {
            "valid_tags": valid_tags, 
            "valid_jbs": valid_jbs,
            "invalid_tags": [item for item in detected_data.get("tags", []) 
                           if item not in valid_tags],
            "invalid_jbs": [item for item in detected_data.get("jbs", []) 
                          if item not in valid_jbs]
        }
    
    def create_jb_tag_summary(self, output_path: str) -> pd.DataFrame:
        """Create and save summary of JB-Tag mappings"""
        summary_data = [
            {
                'JB Name': jb_name,
                'Number of Tags': len(tags),
                'Tags': ', '.join(tags)
            }
            for jb_name, tags in self.jb_tag_mapping.items()
        ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(output_path, index=False)
        print(f"Summary saved to {output_path}")
        return summary_df

    def create_detailed_report(self, detected_data: Dict, validated_data: Dict) -> pd.DataFrame:
        """Create detailed report of detections and validations"""
        report_data = []
        
        # Add JB information
        for jb, bbox in detected_data.get('jbs', []):
            status = 'Valid' if (jb, bbox) in validated_data['valid_jbs'] else 'Invalid'
            report_data.append({
                'Type': 'JB',
                'Text': jb,
                'Status': status,
                'Position': f"({bbox[0]}, {bbox[1]})",
                'Expected Tags': len(self.jb_tag_mapping.get(jb, []))
            })
            
        # Add Tag information
        for tag, bbox in detected_data.get('tags', []):
            status = 'Valid' if (tag, bbox) in validated_data['valid_tags'] else 'Invalid'
            connected_jb = next((jb for jb, tags in self.jb_tag_mapping.items() 
                               if tag in tags), 'Unknown')
            report_data.append({
                'Type': 'Tag',
                'Text': tag,
                'Status': status,
                'Position': f"({bbox[0]}, {bbox[1]})",
                'Connected JB': connected_jb
            })
            
        return pd.DataFrame(report_data)
def process_jb_image(image_path: str, excel_path: str, model_path: str) -> Dict[str, Union[str, Dict]]:
    """Main processing function for JB image analysis"""
    try:
        # 1. Load model
        print("Loading model...")
        model = DBNet(pretrained=False)
        
        with open(model_path, 'rb') as f:
            model_data = f.read()
        model_buffer = io.BytesIO(model_data)
        state_dict = torch.load(model_buffer, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully")
        
        # 2. Initialize processor
        processor = JBImageProcessor(model)
        
        # 3. Process image and detect text
        print("Processing image...")
        results = processor.extract_text_with_coords(image_path)
        print(f"Found {len(results['tags'])} tags and {len(results['jbs'])} JBs")
        
        # 4. Create and save annotated image
        print("Creating annotated image...")
        annotated_image = processor.create_output_image(results)
        output_image_path = "detected_boxes.jpg"
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Annotated image saved to: {output_image_path}")
        
        # 5. Display annotated image in Jupyter
        try:
            display_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(15, 10))
            plt.imshow(display_image_rgb)
            plt.axis('off')
            plt.title('Detected Tags and JBs')
            plt.show()
        except Exception as e:
            print(f"Could not display image in Jupyter: {str(e)}")
        
        # 6. Analyze Excel data
        print("\nAnalyzing Excel data...")
        analyzer = DataAnalysisModule(excel_path)
        analyzer.analyze_tags()
        
        # 7. Validate detections
        validated_results = analyzer.validate_detected_text(results)
        print(f"Validation Results:")
        print(f"Valid Tags: {len(validated_results['valid_tags'])}")
        print(f"Valid JBs: {len(validated_results['valid_jbs'])}")
        print(f"Invalid Tags: {len(validated_results['invalid_tags'])}")
        print(f"Invalid JBs: {len(validated_results['invalid_jbs'])}")
        
        # 8. Create reports
        summary_path = "/home/administrator/Projects/DandC/IODB Excel/jb_tag_summary.xlsx"
        summary_df = analyzer.create_jb_tag_summary(summary_path)
        
        detailed_report_df = analyzer.create_detailed_report(results, validated_results)
        detailed_report_path = "/home/administrator/Projects/DandC/IODB Excel/detection_report.xlsx"
        detailed_report_df.to_excel(detailed_report_path, index=False)
        
        # 9. Display summary results
        print("\nSummary Results:")
        display(summary_df)
        print("\nDetailed Detection Report:")
        display(detailed_report_df)
        
        # 10. Save tag column information
        if 'tag_columns' in results:
            print("\nTag Columns Information:")
            for column_center, tags in results['tag_columns'].items():
                print(f"\nColumn at x={column_center}:")
                for tag in tags:
                    print(f"  {tag}")
        
        # 11. Return all outputs
        return {
            'annotated_image_path': output_image_path,
            'summary_excel_path': summary_path,
            'detailed_report_path': detailed_report_path,
            'detections': {
                'tags': results['tags'],
                'jbs': results['jbs']
            },
            'validated_detections': validated_results,
            'tag_columns': results.get('tag_columns', {}),
            'annotated_image': annotated_image
        }
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        raise

def save_visualization(results: Dict[str, Union[str, Dict]], output_dir: str = "./output") -> None:
    """Save visualization outputs"""
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Save annotated image
        if 'annotated_image' in results:
            image_path = f"{output_dir}/annotated_image.jpg"
            cv2.imwrite(image_path, results['annotated_image'])
            print(f"Saved annotated image to {image_path}")
        
        # 2. Create and save detection statistics visualization
        plt.figure(figsize=(10, 6))
        detection_counts = {
            'Total Tags': len(results['detections']['tags']),
            'Valid Tags': len(results['validated_detections']['valid_tags']),
            'Total JBs': len(results['detections']['jbs']),
            'Valid JBs': len(results['validated_detections']['valid_jbs'])
        }
        
        plt.bar(detection_counts.keys(), detection_counts.values(), color=['lightblue', 'blue', 'lightgreen', 'green'])
        plt.title('Detection and Validation Statistics')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detection_stats.png")
        plt.close()
        
        # 3. Save detection coordinates
        detection_data = []
        for tag, bbox in results['detections']['tags']:
            status = 'Valid' if (tag, bbox) in results['validated_detections']['valid_tags'] else 'Invalid'
            detection_data.append({
                'Type': 'Tag',
                'Text': tag,
                'Status': status,
                'X1': bbox[0],
                'Y1': bbox[1],
                'X2': bbox[2],
                'Y2': bbox[3]
            })
        
        for jb, bbox in results['detections']['jbs']:
            status = 'Valid' if (jb, bbox) in results['validated_detections']['valid_jbs'] else 'Invalid'
            detection_data.append({
                'Type': 'JB',
                'Text': jb,
                'Status': status,
                'X1': bbox[0],
                'Y1': bbox[1],
                'X2': bbox[2],
                'Y2': bbox[3]
            })
        
        pd.DataFrame(detection_data).to_csv(f"{output_dir}/detection_coordinates.csv", index=False)
        
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")
        raise