# feret_dataset.py - Color FERET Dataset Preparation
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class FERETDatasetPreparation:
    """
    Prepare Color FERET dataset with masked images for SN-LF training
    """
    def __init__(self, feret_dir, masked_dir):
        """
        feret_dir: Path to Color FERET dataset
        masked_dir: Path to masked versions of FERET images
        """
        self.feret_dir = feret_dir
        self.masked_dir = masked_dir
        self.dataset_info = None
        
    def parse_feret_structure(self):
        """
        Parse Color FERET dataset structure and create mapping
        FERET uses folder structure: subject_ID/image_files
        Only include folders that exist in both directories
        Handles different file extensions between matched images
        """
        dataset_info = []
        
        print(f"Looking for images in: {self.feret_dir}")
        print(f"Looking for masked images in: {self.masked_dir}")
        
        # Check if directories exist
        if not os.path.exists(self.feret_dir):
            print(f"ERROR: FERET directory not found: {self.feret_dir}")
            return pd.DataFrame(columns=['subject_id', 'unmasked_path', 'masked_path', 'filename'])
        
        if not os.path.exists(self.masked_dir):
            print(f"ERROR: Masked directory not found: {self.masked_dir}")
            return pd.DataFrame(columns=['subject_id', 'unmasked_path', 'masked_path', 'filename'])
        
        # Get all subject folders from unmasked directory
        unmasked_subjects = set([d for d in os.listdir(self.feret_dir) 
                                if os.path.isdir(os.path.join(self.feret_dir, d))])
        
        # Get all subject folders from masked directory
        masked_subjects = set([d for d in os.listdir(self.masked_dir) 
                              if os.path.isdir(os.path.join(self.masked_dir, d))])
        
        # Only process subjects that exist in both directories
        common_subjects = unmasked_subjects.intersection(masked_subjects)
        print(f"Found {len(unmasked_subjects)} folders in unmasked directory")
        print(f"Found {len(masked_subjects)} folders in masked directory")
        print(f"Processing {len(common_subjects)} subjects that exist in both directories")
        
        # Walk through the common subjects
        for subject_id in sorted(common_subjects):
            unmasked_subject_path = os.path.join(self.feret_dir, subject_id)
            masked_subject_path = os.path.join(self.masked_dir, subject_id)
            
            # Get all images in the unmasked subject directory
            unmasked_files = [f for f in os.listdir(unmasked_subject_path) 
                             if f.endswith(('.ppm', '.jpg', '.jpeg', '.png', '.tif', '.bmp', '.pgm'))]
            
            # Get all images in the masked subject directory
            masked_files = [f for f in os.listdir(masked_subject_path) 
                           if f.endswith(('.ppm', '.jpg', '.jpeg', '.png', '.tif', '.bmp', '.pgm'))]
            
            # Match files by base name (without extension)
            unmasked_base_names = {os.path.splitext(f)[0]: f for f in unmasked_files}
            masked_base_names = {os.path.splitext(f)[0]: f for f in masked_files}
            
            # Find common base names
            common_base_names = set(unmasked_base_names.keys()).intersection(set(masked_base_names.keys()))
            
            for base_name in common_base_names:
                unmasked_file = unmasked_base_names[base_name]
                masked_file = masked_base_names[base_name]
                
                unmasked_path = os.path.join(unmasked_subject_path, unmasked_file)
                masked_path = os.path.join(masked_subject_path, masked_file)
                
                dataset_info.append({
                    'subject_id': subject_id,
                    'unmasked_path': unmasked_path,
                    'masked_path': masked_path,
                    'filename': unmasked_file
                })
        
        if not dataset_info:
            print("WARNING: No matching image pairs found!")
            print("Check that:")
            print("1. Both directories have the same folder structure")
            print("2. Files in corresponding folders have matching names (ignoring extensions)")
            return pd.DataFrame(columns=['subject_id', 'unmasked_path', 'masked_path', 'filename'])
        
        self.dataset_info = pd.DataFrame(dataset_info)
        print(f"Found {len(self.dataset_info)} image pairs")
        return self.dataset_info
    
    def create_identity_mapping(self):
        """
        Create numeric identity labels for each subject
        """
        if self.dataset_info is None:
            self.parse_feret_structure()
        
        # Check if dataset_info is empty
        if self.dataset_info is None or self.dataset_info.empty:
            print("ERROR: No data found to create identity mapping!")
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['subject_id', 'image_path', 'identity_label', 'is_masked'])
        
        # Map subject IDs to numeric labels
        unique_subjects = self.dataset_info['subject_id'].unique()
        subject_to_label = {subject: idx for idx, subject in enumerate(unique_subjects)}
        
        self.dataset_info['identity_label'] = self.dataset_info['subject_id'].map(subject_to_label)
        
        # Create combined dataset with both masked and unmasked
        masked_df = self.dataset_info[['subject_id', 'masked_path', 'identity_label']].copy()
        masked_df.columns = ['subject_id', 'image_path', 'identity_label']
        masked_df['is_masked'] = True
        
        unmasked_df = self.dataset_info[['subject_id', 'unmasked_path', 'identity_label']].copy()
        unmasked_df.columns = ['subject_id', 'image_path', 'identity_label']
        unmasked_df['is_masked'] = False
        
        self.combined_dataset = pd.concat([masked_df, unmasked_df], ignore_index=True)
        print(f"Total images (masked + unmasked): {len(self.combined_dataset)}")
        print(f"Number of unique subjects: {self.combined_dataset['identity_label'].nunique()}")
        
        return self.combined_dataset
    
    def create_subset(self, num_images=700):
        """
        Create a balanced subset of the dataset
        """
        if not hasattr(self, 'combined_dataset'):
            self.create_identity_mapping()
        
        # Create balanced subset (equal masked/unmasked)
        masked_subset = self.combined_dataset[self.combined_dataset['is_masked'] == True].sample(
            n=num_images // 2, random_state=42
        )
        unmasked_subset = self.combined_dataset[self.combined_dataset['is_masked'] == False].sample(
            n=num_images // 2, random_state=42
        )
        
        subset_dataset = pd.concat([masked_subset, unmasked_subset])
        
        # Ensure we have multiple images per identity
        identity_counts = subset_dataset['identity_label'].value_counts()
        good_identities = identity_counts[identity_counts >= 2].index
        subset_dataset = subset_dataset[subset_dataset['identity_label'].isin(good_identities)]
        
        print(f"Created subset with {len(subset_dataset)} images")
        print(f"Subjects with multiple images: {len(good_identities)}")
        
        return subset_dataset
    
    def create_train_val_test_split(self, dataset=None, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train, validation, and test sets
        Stratify by subject to ensure all sets have different people
        """
        if dataset is None:
            dataset = self.combined_dataset
        
        # Get unique subjects
        unique_subjects = dataset['subject_id'].unique()
        
        # Split subjects into train, val, test
        train_subjects, test_subjects = train_test_split(
            unique_subjects, test_size=test_ratio, random_state=42
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=val_ratio/(1-test_ratio), random_state=42
        )
        
        # Create splits
        train_data = dataset[dataset['subject_id'].isin(train_subjects)]
        val_data = dataset[dataset['subject_id'].isin(val_subjects)]
        test_data = dataset[dataset['subject_id'].isin(test_subjects)]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data

# For testing/debugging
if __name__ == "__main__":
    # Test with dummy paths (update these to your actual paths)
    FERET_DIR = 'path/to/color_feret'
    MASKED_DIR = 'path/to/color_feret_masked'
    
    # Initialize dataset preparation
    dataset_prep = FERETDatasetPreparation(FERET_DIR, MASKED_DIR)
    
    # Parse dataset structure
    dataset_info = dataset_prep.parse_feret_structure()
    
    # Create identity mapping
    combined_dataset = dataset_prep.create_identity_mapping()
    
    # Create subset
    subset = dataset_prep.create_subset(700)
    
    # Create splits
    train_data, val_data, test_data = dataset_prep.create_train_val_test_split(subset)
    
    # Save datasets for later use
    train_data.to_csv('train_data_700.csv', index=False)
    val_data.to_csv('val_data_700.csv', index=False)
    test_data.to_csv('test_data_700.csv', index=False)
    
    print("Dataset preparation complete. Files saved.")