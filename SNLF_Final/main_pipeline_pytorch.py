# main_pipeline_pytorch.py - Updated main script to use all available images
import os
import sys

# Import our PyTorch modules
from feret_dataset import FERETDatasetPreparation
# Change this import to use the improved trainer
from improved_train_snlf import ImprovedTrainer as OptimizedTrainer

def check_dependencies():
    """Check if all required libraries are installed"""
    required_libraries = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV',
        'skimage': 'scikit-image',
        'matplotlib': 'Matplotlib'
    }
    
    missing_libraries = []
    
    for module, name in required_libraries.items():
        try:
            __import__(module)
        except ImportError:
            missing_libraries.append(name)
    
    if missing_libraries:
        print("The following required libraries are missing:")
        for lib in missing_libraries:
            print(f"- {lib}")
        print("\nPlease install them using:")
        print("pip install -r pytorch_requirements.txt")
        sys.exit(1)
    else:
        print("All required libraries are installed.")

def main():
    """Main pipeline execution"""
    # Check dependencies
    check_dependencies()
    
    # Configuration
    FERET_DIR = '/projectnb/cs585bp/students/dlgirija/colorferet/images_extracted'
    MASKED_DIR = '/projectnb/cs585bp/students/dlgirija/colorferet/random_texture_ff_mask_images'
    # Use all available images instead of limiting to 700
    NUM_IMAGES = None  # None means use all available images
    
    # Output files - keep original paths
    TRAIN_DATA_PATH = 'train_data_full.csv'
    VAL_DATA_PATH = 'val_data_full.csv'
    TEST_DATA_PATH = 'test_data_full.csv'
    MODEL_SAVE_PATH = 'model_full_dataset_pytorch.pth'
    
    # Increased epochs
    EPOCHS = 100  # Doubled from 50 to 100
    
    print("=== Starting SN-LF PyTorch Training Pipeline ===")
    
    # Step 1: Prepare dataset
    print("\n[Step 1] Preparing dataset with all available images...")
    dataset_prep = FERETDatasetPreparation(FERET_DIR, MASKED_DIR)
    
    # Parse dataset structure
    dataset_info = dataset_prep.parse_feret_structure()
    
    # Create identity mapping
    combined_dataset = dataset_prep.create_identity_mapping()
    
    # Create subset or use all data
    if NUM_IMAGES is None:
        # Use all available data
        print("Using all available data")
        subset = combined_dataset
    else:
        # Create subset with specified number of images
        subset = dataset_prep.create_subset(NUM_IMAGES)
    
    # Create splits - ensure different subjects in train/test
    train_data, val_data, test_data = dataset_prep.create_train_val_test_split(subset)
    
    # Save datasets
    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    val_data.to_csv(VAL_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)
    
    print("Dataset preparation complete.")
    
    # Step 2: Train model
    print("\n[Step 2] Training model with improved implementation...")
    trainer = OptimizedTrainer(MODEL_SAVE_PATH, epochs=EPOCHS)
    
    # Train the model
    history = trainer.train(TRAIN_DATA_PATH, VAL_DATA_PATH)
    
    print("\n=== Training Pipeline Complete ===")
    print(f"- Dataset files: {TRAIN_DATA_PATH}, {VAL_DATA_PATH}, {TEST_DATA_PATH}")
    print(f"- Model weights: {MODEL_SAVE_PATH}")
    print(f"- Training history plot: training_history_improved.png")
    
    # Note about testing
    print("\nNext steps:")
    print("1. Use test_accuracy_pytorch.py to evaluate the model on the test set")
    print("2. Use predict_snlf_pytorch.py to make predictions on new face pairs")

if __name__ == "__main__":
    main()