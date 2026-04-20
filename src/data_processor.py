"""
Dataset Processing Module
Responsible for scanning datasets, generating indices, sampling, and related functionalities.
"""
import json
import os
import random
from pathlib import Path
from src.utils import get_label_from_filename, ensure_dir


class DatasetProcessor:
    def __init__(self, images_dir, index_file="data/dataset_index.json"):
        """
        Initialize the dataset processor.
        
        Args:
            images_dir: Path to the directory containing images.
            index_file: Path to the dataset index file.
        """
        self.images_dir = Path(images_dir)
        self.index_file = Path(index_file)
        self.dataset_index = None
        
    def build_index(self, force_rebuild=False):
        """
        Build the dataset index by scanning all images and extracting labels.
        
        Args:
            force_rebuild: Whether to force rebuilding the index (even if it already exists).
            
        Returns:
            Dictionary containing the dataset index.
        """
        # If index exists and force_rebuild is False, load it directly
        if self.index_file.exists() and not force_rebuild:
            print(f"Index file already exists: {self.index_file}")
            print("Loading existing index...")
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.dataset_index = json.load(f)
            print(f"Loaded {len(self.dataset_index['images'])} images")
            return self.dataset_index
        
        print(f"Starting to build dataset index: {self.images_dir}")
        
        images_data = []
        clean_count = 0
        dirty_count = 0
        
        # Scan all jpg files
        for img_file in self.images_dir.glob("*.jpg"):
            try:
                filename = img_file.name
                
                # Skip checkpoint files
                if 'checkpoint' in filename:
                    continue
                
                # Extract label
                label = get_label_from_filename(filename)
                
                # Record image information
                images_data.append({
                    "filename": filename,
                    "path": str(img_file.absolute()),
                    "label": label,
                    "label_name": "clean" if label == 0 else "dirty"
                })
                
                if label == 0:
                    clean_count += 1
                else:
                    dirty_count += 1
                    
            except ValueError as e:
                print(f"Skipping file {img_file.name}: {e}")
                continue
        
        # Construct index structure
        self.dataset_index = {
            "total": len(images_data),
            "clean": clean_count,
            "dirty": dirty_count,
            "images_dir": str(self.images_dir),
            "images": images_data
        }
        
        # Save index
        ensure_dir(self.index_file.parent)
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset_index, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset index build complete!")
        print(f"  Total: {self.dataset_index['total']} images")
        print(f"  Clean: {clean_count} images")
        print(f"  Dirty: {dirty_count} images")
        print(f"  Index saved to: {self.index_file}")
        
        return self.dataset_index
    
    def load_index(self):
        """Load existing dataset index."""
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file does not exist: {self.index_file}. Please run build_index() first.")
        
        with open(self.index_file, 'r', encoding='utf-8') as f:
            self.dataset_index = json.load(f)
        
        return self.dataset_index
    
    def sample_dataset(self, n_samples, balanced=True, random_seed=42):
        """
        Sample a specified number of images from the dataset.
        
        Args:
            n_samples: Number of samples to draw.
            balanced: Whether to use balanced sampling (equal number per class).
            random_seed: Random seed for reproducibility.
            
        Returns:
            List of sampled image dictionaries.
        """
        if self.dataset_index is None:
            self.load_index()
        
        random.seed(random_seed)
        
        if balanced:
            # Balanced sampling
            samples_per_class = n_samples // 2
            
            # Categorize images
            clean_images = [img for img in self.dataset_index['images'] if img['label'] == 0]
            dirty_images = [img for img in self.dataset_index['images'] if img['label'] == 1]
            
            # Check for sufficient samples
            if len(clean_images) < samples_per_class or len(dirty_images) < samples_per_class:
                print(f"Warning: Requested {samples_per_class} per class, but dataset only contains:")
                print(f"  Clean: {len(clean_images)}")
                print(f"  Dirty: {len(dirty_images)}")
                samples_per_class = min(len(clean_images), len(dirty_images), samples_per_class)
            
            # Random sampling per class
            sampled_clean = random.sample(clean_images, samples_per_class)
            sampled_dirty = random.sample(dirty_images, samples_per_class)
            
            sampled_images = sampled_clean + sampled_dirty
            random.shuffle(sampled_images)  # Shuffle the combined list
            
            print(f"Balanced sampling complete: {len(sampled_images)} images")
            print(f"  Clean: {samples_per_class}")
            print(f"  Dirty: {samples_per_class}")
            
        else:
            # Simple random sampling
            if n_samples > len(self.dataset_index['images']):
                print(f"Warning: Requested {n_samples} samples, but dataset only has {len(self.dataset_index['images'])}")
                n_samples = len(self.dataset_index['images'])
            
            sampled_images = random.sample(self.dataset_index['images'], n_samples)
            
            print(f"Random sampling complete: {len(sampled_images)} images")
        
        return sampled_images
    
    def get_statistics(self):
        """Get dataset statistical information."""
        if self.dataset_index is None:
            self.load_index()
        
        return {
            "total": self.dataset_index["total"],
            "clean": self.dataset_index["clean"],
            "dirty": self.dataset_index["dirty"],
            "balance_ratio": self.dataset_index["clean"] / self.dataset_index["dirty"] if self.dataset_index["dirty"] > 0 else 0
        }


if __name__ == "__main__":
    # Test code
    processor = DatasetProcessor("dataset/archive/solar_panel_dust_segmentation/images")
    
    # Build index
    processor.build_index()
    
    # Get statistics
    stats = processor.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Clean: {stats['clean']}")
    print(f"  Dirty: {stats['dirty']}")
    print(f"  Balance Ratio: {stats['balance_ratio']:.2f}")
    
    # Test sampling
    print("\nTesting sampling for 10 images:")
    samples = processor.sample_dataset(10, balanced=True)
    for sample in samples[:5]:
        print(f"  {sample['filename']} -> {sample['label_name']}")