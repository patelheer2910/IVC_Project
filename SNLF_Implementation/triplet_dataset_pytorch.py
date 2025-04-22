# triplet_dataset_pytorch.py - PyTorch Dataset for Triplet Generation
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import pandas as pd

class FERETTripletDataset(Dataset):
    """
    PyTorch Dataset for FERET triplet generation
    """
    def __init__(self, dataframe, image_size=(32, 32), lbp_processor=None):
        self.dataframe = dataframe
        self.image_size = image_size
        self.lbp_processor = lbp_processor
        
        # Group by identity
        self.identity_groups = dataframe.groupby('identity_label')
        self.identity_labels = list(self.identity_groups.groups.keys())
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Select anchor identity
        anchor_identity = random.choice(self.identity_labels)
        anchor_group = self.identity_groups.get_group(anchor_identity)
        
        # Sample anchor and positive (same person)
        if len(anchor_group) >= 2:
            anchor_row, positive_row = anchor_group.sample(2, replace=False).iloc[0], anchor_group.sample(2, replace=False).iloc[1]
        else:
            anchor_row = anchor_group.sample(1).iloc[0]
            positive_row = anchor_row  # Use same image as positive if only one available
        
        # Sample negative (different person)
        negative_identity = random.choice([id for id in self.identity_labels if id != anchor_identity])
        negative_group = self.identity_groups.get_group(negative_identity)
        negative_row = negative_group.sample(1).iloc[0]
        
        # Load and preprocess images
        anchor_img = self._load_and_preprocess(anchor_row['image_path'])
        positive_img = self._load_and_preprocess(positive_row['image_path'])
        negative_img = self._load_and_preprocess(negative_row['image_path'])
        
        # Process with LBP if processor is provided
        if self.lbp_processor:
            anchor_img = self.lbp_processor.process_image(anchor_img)
            positive_img = self.lbp_processor.process_image(positive_img)
            negative_img = self.lbp_processor.process_image(negative_img)
        
        # Convert to tensors
        anchor_tensor = torch.from_numpy(anchor_img).float().unsqueeze(0) / 255.0
        positive_tensor = torch.from_numpy(positive_img).float().unsqueeze(0) / 255.0
        negative_tensor = torch.from_numpy(negative_img).float().unsqueeze(0) / 255.0
        
        return anchor_tensor, positive_tensor, negative_tensor
    
    def _load_and_preprocess(self, image_path):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img

# For testing/debugging
if __name__ == "__main__":
    # Test dataset
    sample_data = pd.DataFrame({
        'image_path': ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg'],
        'identity_label': [0, 0, 1],
        'subject_id': ['person1', 'person1', 'person2'],
        'is_masked': [True, False, True]
    })
    
    dataset = FERETTripletDataset(sample_data)
    print(f"Dataset size: {len(dataset)}")