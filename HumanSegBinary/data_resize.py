import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

class ImagePreprocessor:
    def __init__(self, source_dir, output_dir, target_size=(640, 480), train_ratio=0.8):
        """
        Initialize the Image Preprocessor
        
        Args:
            source_dir (str): Path to source directory containing images
            output_dir (str): Path to output directory for processed data
            target_size (tuple): Target dimensions (width, height)
            train_ratio (float): Ratio of training data (0.8 = 80%)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.train_ratio = train_ratio
        
        # Define paths
        self.images_dir = self.source_dir / "Training_Images"
        self.masks_dir = self.source_dir / "Ground_Truth"
        
        # Create output directory structure
        self.setup_output_dirs()
    
    def setup_output_dirs(self):
        """Create the output directory structure"""
        dirs_to_create = [
            self.output_dir / "train" / "images",
            self.output_dir / "train" / "masks",
            self.output_dir / "test" / "images", 
            self.output_dir / "test" / "masks"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def resize_image_direct(self, image, target_size):
        """
        Resize image directly to target dimensions (like WordPad/image viewers)
        
        Args:
            image: Input image array
            target_size: Target (width, height)
            
        Returns:
            Resized image to exact target dimensions
        """
        target_w, target_h = target_size
        
        # Direct resize to exact target dimensions
        if len(image.shape) == 3:
            # For color images, use high-quality interpolation
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            # For masks, use nearest neighbor to preserve discrete values
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
        return resized
    
    def get_image_pairs(self):
        """
        Get pairs of images and their corresponding masks
        
        Returns:
            List of tuples (image_path, mask_path)
        """
        image_files = []
        mask_files = []
        
        # Get all image files (assuming they're named 1.jpg to 300.jpg)
        for i in range(1, 301):
            img_path = self.images_dir / f"{i}.jpg"
            mask_path = self.masks_dir / f"{i}.png"  # Assuming masks are PNG
            
            if img_path.exists() and mask_path.exists():
                image_files.append(img_path)
                mask_files.append(mask_path)
            else:
                # Try different extensions
                for img_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    for mask_ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        img_path_alt = self.images_dir / f"{i}{img_ext}"
                        mask_path_alt = self.masks_dir / f"{i}{mask_ext}"
                        
                        if img_path_alt.exists() and mask_path_alt.exists():
                            image_files.append(img_path_alt)
                            mask_files.append(mask_path_alt)
                            break
                    else:
                        continue
                    break
        
        return list(zip(image_files, mask_files))
    
    def process_and_save_image(self, image_path, mask_path, output_img_path, output_mask_path):
        """
        Process a single image-mask pair and save to output directory
        
        Args:
            image_path: Path to source image
            mask_path: Path to source mask
            output_img_path: Path to save processed image
            output_mask_path: Path to save processed mask
        """
        try:
            # Load image and mask
            image = cv2.imread(str(image_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Warning: Could not load {image_path} or {mask_path}")
                return False
            
            # Resize both image and mask to exact target dimensions
            processed_image = self.resize_image_direct(image, self.target_size)
            processed_mask = self.resize_image_direct(mask, self.target_size)
            
            # Save processed files
            cv2.imwrite(str(output_img_path), processed_image, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(output_mask_path), processed_mask, 
                       [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False
    
    def split_and_process_data(self):
        """
        Split data into train/test and process all images
        """
        # Get all image-mask pairs
        image_mask_pairs = self.get_image_pairs()
        
        if not image_mask_pairs:
            print("No image-mask pairs found! Please check your directory structure.")
            return
        
        print(f"Found {len(image_mask_pairs)} image-mask pairs")
        
        # Split into train and test
        train_pairs, test_pairs = train_test_split(
            image_mask_pairs, 
            train_size=self.train_ratio, 
            random_state=42,
            shuffle=True
        )
        
        print(f"Train set: {len(train_pairs)} pairs")
        print(f"Test set: {len(test_pairs)} pairs")
        
        # Process training data
        print("\nProcessing training data...")
        train_success = 0
        for idx, (img_path, mask_path) in enumerate(tqdm(train_pairs, desc="Training")):
            output_img = self.output_dir / "train" / "images" / f"train_{idx+1:04d}.jpg"
            output_mask = self.output_dir / "train" / "masks" / f"train_{idx+1:04d}.png"
            
            if self.process_and_save_image(img_path, mask_path, output_img, output_mask):
                train_success += 1
        
        # Process test data
        print("\nProcessing test data...")
        test_success = 0
        for idx, (img_path, mask_path) in enumerate(tqdm(test_pairs, desc="Testing")):
            output_img = self.output_dir / "test" / "images" / f"test_{idx+1:04d}.jpg"
            output_mask = self.output_dir / "test" / "masks" / f"test_{idx+1:04d}.png"
            
            if self.process_and_save_image(img_path, mask_path, output_img, output_mask):
                test_success += 1
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed {train_success}/{len(train_pairs)} training pairs")
        print(f"Successfully processed {test_success}/{len(test_pairs)} test pairs")
        
        # Create summary file
        self.create_summary_file(train_success, test_success, len(train_pairs), len(test_pairs))
    
    def create_summary_file(self, train_success, test_success, total_train, total_test):
        """Create a summary file with processing statistics"""
        summary_path = self.output_dir / "processing_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Image Preprocessing Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Target Image Size: {self.target_size[0]} x {self.target_size[1]}\n")
            f.write(f"Train Ratio: {self.train_ratio * 100}%\n\n")
            f.write(f"Training Data:\n")
            f.write(f"  - Total pairs: {total_train}\n")
            f.write(f"  - Successfully processed: {train_success}\n")
            f.write(f"  - Success rate: {train_success/total_train*100:.1f}%\n\n")
            f.write(f"Test Data:\n")
            f.write(f"  - Total pairs: {total_test}\n")
            f.write(f"  - Successfully processed: {test_success}\n")
            f.write(f"  - Success rate: {test_success/total_test*100:.1f}%\n\n")
            f.write("Directory Structure:\n")
            f.write("  processed_data/\n")
            f.write("  ├── train/\n")
            f.write("  │   ├── images/\n")
            f.write("  │   └── masks/\n")
            f.write("  └── test/\n")
            f.write("      ├── images/\n")
            f.write("      └── masks/\n")
        
        print(f"\nSummary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess human segmentation dataset')
    parser.add_argument('--source', '-s', type=str, 
                       default='data/Human-Segmentation-Dataset',
                       help='Source directory containing Training_Images and Ground_Truth')
    parser.add_argument('--output', '-o', type=str, 
                       default='data/processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--width', '-w', type=int, default=320,
                       help='Target width (default: 640)')
    parser.add_argument('--height', '--ht', type=int, default=240,
                       help='Target height (default: 480)')
    parser.add_argument('--train-ratio', '-r', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        source_dir=args.source,
        output_dir=args.output,
        target_size=(args.width, args.height),
        train_ratio=args.train_ratio
    )
    
    # Process the data
    preprocessor.split_and_process_data()

if __name__ == "__main__":
    main()