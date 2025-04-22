# test_accuracy_pytorch.py - Compute accuracy on unseen images
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from snlf_model_pytorch import MaskedFaceDetector, LBPProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class TestDataset(Dataset):
    """Dataset for testing accuracy on unseen images"""
    def __init__(self, test_df, image_size=(32, 32)):
        self.df = test_df
        self.image_size = image_size
        self.lbp_processor = LBPProcessor()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and preprocess image
        img = cv2.imread(row['image_path'])
        if img is None:
            raise ValueError(f"Could not load image: {row['image_path']}")
        
        # Resize and convert to grayscale
        img = cv2.resize(img, self.image_size)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Process with LBP
        img_lbp = self.lbp_processor.process_image(img)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_lbp).float().unsqueeze(0) / 255.0
        
        return img_tensor, row['identity_label'], row['subject_id']

def predict_pair(detector, img1_path, img2_path, threshold=0.5):
    """Properly predict a pair of images"""
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Ensure images are loaded
    if img1 is None or img2 is None:
        print(f"Error loading images: {img1_path} or {img2_path}")
        return False, 999.0
    
    # Resize to expected dimensions
    img1 = cv2.resize(img1, (32, 32))
    img2 = cv2.resize(img2, (32, 32))
    
    # Preprocess with LBP
    img1_lbp = detector.lbp_processor.process_image(img1)
    img2_lbp = detector.lbp_processor.process_image(img2)
    
    # Convert to tensor
    img1_tensor = torch.from_numpy(img1_lbp).float().unsqueeze(0).unsqueeze(0).to(detector.device) / 255.0
    img2_tensor = torch.from_numpy(img2_lbp).float().unsqueeze(0).unsqueeze(0).to(detector.device) / 255.0
    
    # Get embeddings
    with torch.no_grad():
        emb1 = detector.siamese_net(img1_tensor)
        emb2 = detector.siamese_net(img2_tensor)
    
    # Compute distance
    distance = torch.norm(emb1 - emb2, dim=1).item()
    
    # Check if same person
    is_same = distance < threshold
    
    return is_same, distance

def compute_accuracy(model_path, test_csv_path, threshold=0.5):
    """
    Compute accuracy on unseen test images
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    detector = MaskedFaceDetector(model_path=model_path, device=device)
    detector.siamese_net.eval()
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    
    # Generate pairs for testing
    print("Generating test pairs...")
    test_pairs = generate_test_pairs(test_df, num_pairs=1000)
    
    # Evaluate pairs
    true_labels = []
    predicted_labels = []
    distances = []
    
    print("Evaluating pairs...")
    for pair in tqdm(test_pairs):
        img1_path = pair['img1_path']
        img2_path = pair['img2_path']
        is_same = pair['is_same']
        
        # Make prediction
        predicted_same, distance = predict_pair(detector, img1_path, img2_path, threshold=threshold)
        
        true_labels.append(is_same)
        predicted_labels.append(predicted_same)
        distances.append(distance)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    print("\n=== Test Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot distance distribution
    same_distances = [d for d, is_same in zip(distances, true_labels) if is_same]
    diff_distances = [d for d, is_same in zip(distances, true_labels) if not is_same]
    
    plt.figure(figsize=(10, 6))
    plt.hist(same_distances, bins=50, alpha=0.5, label='Same Person', color='green')
    plt.hist(diff_distances, bins=50, alpha=0.5, label='Different People', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold={threshold}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution for Same vs Different People')
    plt.legend()
    plt.savefig('test_distance_distribution.png')
    plt.show()
    
    return accuracy, precision, recall, f1, distances

def generate_test_pairs(test_df, num_pairs=1000):
    """Generate balanced pairs of same/different people for testing"""
    pairs = []
    identity_labels = test_df['identity_label'].unique()
    
    # Generate same-person pairs (50%)
    for _ in range(num_pairs // 2):
        # Select a person with at least 2 images
        valid_labels = [label for label in identity_labels 
                       if len(test_df[test_df['identity_label'] == label]) >= 2]
        if not valid_labels:
            continue
            
        label = random.choice(valid_labels)
        person_df = test_df[test_df['identity_label'] == label]
        
        # Select two different images
        imgs = person_df.sample(2)
        pairs.append({
            'img1_path': imgs.iloc[0]['image_path'],
            'img2_path': imgs.iloc[1]['image_path'],
            'is_same': True
        })
    
    # Generate different-person pairs (50%)
    for _ in range(num_pairs // 2):
        # Select two different people
        if len(identity_labels) < 2:
            continue
            
        label1, label2 = random.sample(list(identity_labels), 2)
        
        person1_img = test_df[test_df['identity_label'] == label1].sample(1).iloc[0]
        person2_img = test_df[test_df['identity_label'] == label2].sample(1).iloc[0]
        
        pairs.append({
            'img1_path': person1_img['image_path'],
            'img2_path': person2_img['image_path'],
            'is_same': False
        })
    
    return pairs

def find_optimal_threshold(model_path, test_csv_path):
    """Find the optimal threshold for classification"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = MaskedFaceDetector(model_path=model_path, device=device)
    detector.siamese_net.eval()
    
    test_df = pd.read_csv(test_csv_path)
    test_pairs = generate_test_pairs(test_df, num_pairs=500)
    
    distances = []
    labels = []
    
    print("Computing distances for threshold optimization...")
    for pair in tqdm(test_pairs):
        # Use the fixed prediction function
        _, distance = predict_pair(detector, pair['img1_path'], pair['img2_path'])
        
        distances.append(distance)
        labels.append(pair['is_same'])
    
    # Try different thresholds
    thresholds = np.linspace(min(distances), max(distances), 100)
    best_accuracy = 0
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = [d < threshold for d in distances]
        accuracy = accuracy_score(labels, predictions)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.4f}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    return best_threshold

if __name__ == "__main__":
    # Paths
    MODEL_PATH = 'model_700_images_pytorch.pth'
    TEST_CSV_PATH = 'test_data_700.csv'
    
    # Find optimal threshold
    print("Finding optimal threshold...")
    optimal_threshold = find_optimal_threshold(MODEL_PATH, TEST_CSV_PATH)
    
    # Compute accuracy with optimal threshold
    print("\nComputing accuracy with optimal threshold...")
    accuracy, precision, recall, f1, distances = compute_accuracy(
        MODEL_PATH, 
        TEST_CSV_PATH, 
        threshold=optimal_threshold
    )
    
    # Also compute with default threshold for comparison
    print("\nComputing accuracy with default threshold (0.5)...")
    accuracy_default, precision_default, recall_default, f1_default, _ = compute_accuracy(
        MODEL_PATH, 
        TEST_CSV_PATH, 
        threshold=0.5
    )
    
    print("\n=== Comparison ===")
    print(f"Optimal threshold ({optimal_threshold:.4f}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\nDefault threshold (0.5):")
    print(f"  Accuracy: {accuracy_default:.4f}")
    print(f"  F1 Score: {f1_default:.4f}")