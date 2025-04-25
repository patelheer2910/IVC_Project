# triplet_dataset_pytorch.py - PyTorch Dataset for Triplet Generation
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import pandas as pd

class FERETTripletDataset(Dataset):
    """
    PyTorch Dataset for FERET triplet generation with enhanced sampling
    """
    def __init__(self, dataframe, image_size=(32, 32), lbp_processor=None, augment=True):
        self.dataframe = dataframe
        self.image_size = image_size
        self.lbp_processor = lbp_processor
        self.augment = augment
        
        # Group by identity
        self.identity_groups = dataframe.groupby('identity_label')
        self.identity_labels = list(self.identity_groups.groups.keys())
        
        # Create separate lookups for masked and unmasked images
        self.masked_lookup = {}
        self.unmasked_lookup = {}
        
        for identity in self.identity_labels:
            group = self.identity_groups.get_group(identity)
            
            # Get masked and unmasked images for this identity
            masked = group[group['is_masked'] == True]
            unmasked = group[group['is_masked'] == False]
            
            if not masked.empty:
                self.masked_lookup[identity] = masked.index.tolist()
            if not unmasked.empty:
                self.unmasked_lookup[identity] = unmasked.index.tolist()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Use a smarter sampling strategy (4 different types of triplets)
        sampling_type = np.random.randint(0, 4)
        
        # Select anchor identity from available identities
        anchor_identity = np.random.choice(self.identity_labels)
        
        # Sample anchor based on sampling type
        anchor_idx = self._sample_anchor(anchor_identity, sampling_type)
        anchor_row = self.dataframe.iloc[anchor_idx]
        
        # Sample positive (same identity as anchor)
        positive_idx = self._sample_positive(anchor_identity, anchor_idx, sampling_type)
        positive_row = self.dataframe.iloc[positive_idx]
        
        # Sample negative (different identity)
        # For harder negatives, try to pick similar looking people when possible
        negative_idx = self._sample_negative(anchor_identity, sampling_type)
        negative_row = self.dataframe.iloc[negative_idx]
        
        # Load and preprocess images
        anchor_img = self._load_and_preprocess(anchor_row['image_path'], is_anchor=True)
        positive_img = self._load_and_preprocess(positive_row['image_path'], is_anchor=False)
        negative_img = self._load_and_preprocess(negative_row['image_path'], is_anchor=False)
        
        return anchor_img, positive_img, negative_img
    
    def _sample_anchor(self, identity, sampling_type):
        """Sample anchor image based on sampling type"""
        # Type 0, 2: unmasked anchor; Type 1, 3: masked anchor
        if sampling_type in [0, 2]:
            # Unmasked anchor
            if identity in self.unmasked_lookup and self.unmasked_lookup[identity]:
                anchor_idx = np.random.choice(self.unmasked_lookup[identity])
            else:
                # Fallback to masked if no unmasked available
                anchor_idx = np.random.choice(self.masked_lookup[identity])
        else:
            # Masked anchor
            if identity in self.masked_lookup and self.masked_lookup[identity]:
                anchor_idx = np.random.choice(self.masked_lookup[identity])
            else:
                # Fallback to unmasked if no masked available
                anchor_idx = np.random.choice(self.unmasked_lookup[identity])
        
        return anchor_idx
    
    def _sample_positive(self, identity, anchor_idx, sampling_type):
        """Sample positive image (same identity as anchor)"""
        # Type 0, 3: different mask type than anchor; Type 1, 2: same mask type
        anchor_is_masked = self.dataframe.iloc[anchor_idx]['is_masked']
        
        if sampling_type in [0, 3]:
            # Different mask type (masked vs unmasked)
            if anchor_is_masked:
                # Anchor is masked, positive should be unmasked
                if identity in self.unmasked_lookup and self.unmasked_lookup[identity]:
                    candidates = self.unmasked_lookup[identity]
                else:
                    # Fallback to same mask type if necessary
                    candidates = [idx for idx in self.masked_lookup[identity] if idx != anchor_idx]
                    if not candidates:
                        # If no other images, use anchor as fallback
                        return anchor_idx
            else:
                # Anchor is unmasked, positive should be masked
                if identity in self.masked_lookup and self.masked_lookup[identity]:
                    candidates = self.masked_lookup[identity]
                else:
                    # Fallback to same mask type
                    candidates = [idx for idx in self.unmasked_lookup[identity] if idx != anchor_idx]
                    if not candidates:
                        # If no other images, use anchor as fallback
                        return anchor_idx
        else:
            # Same mask type
            if anchor_is_masked:
                # Both masked
                candidates = [idx for idx in self.masked_lookup.get(identity, []) if idx != anchor_idx]
                if not candidates:
                    # Fallback to different mask type
                    candidates = self.unmasked_lookup.get(identity, [])
                    if not candidates:
                        # If no alternatives, use anchor as fallback
                        return anchor_idx
            else:
                # Both unmasked
                candidates = [idx for idx in self.unmasked_lookup.get(identity, []) if idx != anchor_idx]
                if not candidates:
                    # Fallback to different mask type
                    candidates = self.masked_lookup.get(identity, [])
                    if not candidates:
                        # If no alternatives, use anchor as fallback
                        return anchor_idx
        
        positive_idx = np.random.choice(candidates)
        return positive_idx
    
    def _sample_negative(self, anchor_identity, sampling_type):
        """Sample negative image (different identity)"""
        # Choose a different identity
        other_identities = [id for id in self.identity_labels if id != anchor_identity]
        negative_identity = np.random.choice(other_identities)
        
        # Type 0, 2: unmasked negative; Type 1, 3: masked negative
        if sampling_type in [0, 2]:
            # Unmasked negative
            if negative_identity in self.unmasked_lookup and self.unmasked_lookup[negative_identity]:
                negative_idx = np.random.choice(self.unmasked_lookup[negative_identity])
            else:
                # Fallback to masked
                negative_idx = np.random.choice(self.masked_lookup[negative_identity])
        else:
            # Masked negative
            if negative_identity in self.masked_lookup and self.masked_lookup[negative_identity]:
                negative_idx = np.random.choice(self.masked_lookup[negative_identity])
            else:
                # Fallback to unmasked
                negative_idx = np.random.choice(self.unmasked_lookup[negative_identity])
        
        return negative_idx
    
    def _load_and_preprocess(self, image_path, is_anchor=False):
        """Load and preprocess image with optional augmentation"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply data augmentation (only for non-anchor images to make triplet harder)
        if self.augment and not is_anchor:
            img = self._apply_augmentation(img)
        
        # Apply LBP processing if processor provided
        if self.lbp_processor:
            img = self.lbp_processor.process_image(img)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        return img_tensor
    
    def _apply_augmentation(self, img):
        """Apply random augmentation to make triplets more challenging"""
        # Only apply augmentation sometimes (50% chance)
        if np.random.random() > 0.5:
            return img
        
        # Random rotation (slight)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)  # 1 means horizontal flip
        
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