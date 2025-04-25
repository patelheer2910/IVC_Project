# improved_test_accuracy.py - Enhanced accuracy testing for SN-LF
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our improved model
from improved_snlf_model import MaskedFaceDetector

class TestPairDataset(Dataset):
    """
    Dataset for testing with pairs of images (same/different people)
    """
    def __init__(self, test_pairs):
        self.test_pairs = test_pairs
    
    def __len__(self):
        return len(self.test_pairs)
    
    def __getitem__(self, idx):
        pair = self.test_pairs[idx]
        
        # Return image paths and label (is_same)
        return pair['img1_path'], pair['img2_path'], pair['is_same']

def generate_test_pairs(test_df, num_pairs=2000, balanced=True):
    """
    Generate pairs for testing with more diverse sampling as described in paper
    
    Args:
        test_df: DataFrame with test images
        num_pairs: Number of pairs to generate
        balanced: Whether to balance same/different pairs (50/50 split)
        
    Returns:
        List of dictionaries with image pairs and labels
    """
    pairs = []
    identity_groups = test_df.groupby('identity_label')
    identity_labels = list(identity_groups.groups.keys())
    
    # Create lookup for masked/unmasked images
    masked_lookup = {}
    unmasked_lookup = {}
    
    for identity in identity_labels:
        group = identity_groups.get_group(identity)
        
        # Get masked and unmasked images for this identity
        masked = group[group['is_masked'] == True]
        unmasked = group[group['is_masked'] == False]
        
        if not masked.empty:
            masked_lookup[identity] = masked['image_path'].tolist()
        if not unmasked.empty:
            unmasked_lookup[identity] = unmasked['image_path'].tolist()
    
    # Calculate number of pairs for each category
    if balanced:
        num_same_pairs = num_pairs // 2
        num_diff_pairs = num_pairs // 2
    else:
        # Unbalanced: 30% same, 70% different (more challenging test)
        num_same_pairs = int(num_pairs * 0.3)
        num_diff_pairs = num_pairs - num_same_pairs
    
    print(f"Generating {num_same_pairs} same-identity pairs and {num_diff_pairs} different-identity pairs")
    
    # Generate same-identity pairs (with both masked and unmasked combinations)
    valid_identities = [
        identity for identity in identity_labels 
        if (identity in masked_lookup and len(masked_lookup[identity]) > 1) or
           (identity in unmasked_lookup and len(unmasked_lookup[identity]) > 1) or
           (identity in masked_lookup and identity in unmasked_lookup)
    ]
    
    if not valid_identities:
        raise ValueError("No valid identities found with multiple images for same-identity pairs")
    
    # Same-person pairs with various mask combinations
    for _ in range(num_same_pairs):
        # Randomly select a valid identity
        identity = np.random.choice(valid_identities)
        
        # Randomly select mask combination:
        # 0: unmasked-unmasked, 1: masked-masked, 2: unmasked-masked
        mask_combination = np.random.randint(0, 3)
        
        if mask_combination == 0 and identity in unmasked_lookup and len(unmasked_lookup[identity]) >= 2:
            # Unmasked-Unmasked
            img1, img2 = np.random.choice(unmasked_lookup[identity], 2, replace=False)
            
        elif mask_combination == 1 and identity in masked_lookup and len(masked_lookup[identity]) >= 2:
            # Masked-Masked
            img1, img2 = np.random.choice(masked_lookup[identity], 2, replace=False)
            
        elif mask_combination == 2 and identity in unmasked_lookup and identity in masked_lookup:
            # Unmasked-Masked
            img1 = np.random.choice(unmasked_lookup[identity])
            img2 = np.random.choice(masked_lookup[identity])
            
        else:
            # Fallback: use any available images for this identity
            all_images = []
            if identity in unmasked_lookup:
                all_images.extend(unmasked_lookup[identity])
            if identity in masked_lookup:
                all_images.extend(masked_lookup[identity])
                
            if len(all_images) >= 2:
                img1, img2 = np.random.choice(all_images, 2, replace=False)
            else:
                # Skip this iteration if not enough images
                continue
        
        pairs.append({
            'img1_path': img1,
            'img2_path': img2,
            'is_same': True
        })
    
    # Generate different-identity pairs (with challenging combinations)
    for _ in range(num_diff_pairs):
        # Select two different identities
        if len(identity_labels) < 2:
            break
            
        id1, id2 = np.random.choice(identity_labels, 2, replace=False)
        
        # Randomly select mask combination for challenging pairs
        mask_combination = np.random.randint(0, 4)
        
        if mask_combination == 0 and id1 in unmasked_lookup and id2 in unmasked_lookup:
            # Both unmasked
            img1 = np.random.choice(unmasked_lookup[id1])
            img2 = np.random.choice(unmasked_lookup[id2])
            
        elif mask_combination == 1 and id1 in masked_lookup and id2 in masked_lookup:
            # Both masked
            img1 = np.random.choice(masked_lookup[id1])
            img2 = np.random.choice(masked_lookup[id2])
            
        elif mask_combination == 2 and id1 in unmasked_lookup and id2 in masked_lookup:
            # First unmasked, second masked
            img1 = np.random.choice(unmasked_lookup[id1])
            img2 = np.random.choice(masked_lookup[id2])
            
        elif mask_combination == 3 and id1 in masked_lookup and id2 in unmasked_lookup:
            # First masked, second unmasked
            img1 = np.random.choice(masked_lookup[id1])
            img2 = np.random.choice(unmasked_lookup[id2])
            
        else:
            # Fallback: use any available images
            img1 = np.random.choice(test_df[test_df['identity_label'] == id1]['image_path'].values)
            img2 = np.random.choice(test_df[test_df['identity_label'] == id2]['image_path'].values)
        
        pairs.append({
            'img1_path': img1,
            'img2_path': img2,
            'is_same': False
        })
    
    print(f"Generated {len(pairs)} test pairs")
    return pairs

def find_optimal_threshold(model_path, test_csv_path, num_pairs=500):
    """Find the optimal threshold for classification through ROC curve analysis"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = MaskedFaceDetector(model_path=model_path, device=device)
    detector.siamese_net.eval()
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    
    # Generate balanced test pairs
    test_pairs = generate_test_pairs(test_df, num_pairs=num_pairs)
    
    # Create dataset and dataloader
    test_dataset = TestPairDataset(test_pairs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one pair at a time
        shuffle=False,
        num_workers=4
    )
    
    # Collect predictions and labels
    distances = []
    labels = []
    
    print("Computing distances for ROC analysis...")
    for img1_path, img2_path, is_same in tqdm(test_loader):
        # Load and preprocess images
        img1 = cv2.imread(img1_path[0])
        img2 = cv2.imread(img2_path[0])
        
        if img1 is None or img2 is None:
            print(f"Error loading images: {img1_path[0]} or {img2_path[0]}")
            continue
        
        # Preprocess with LBP
        img1_lbp = detector.lbp_processor.process_image(img1)
        img2_lbp = detector.lbp_processor.process_image(img2)
        
        # Convert to tensor
        img1_tensor = torch.from_numpy(img1_lbp).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        img2_tensor = torch.from_numpy(img2_lbp).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        
        # Get embeddings
        with torch.no_grad():
            emb1 = detector.siamese_net(img1_tensor)
            emb2 = detector.siamese_net(img2_tensor)
        
        # Compute distance (smaller = more similar)
        distance = torch.norm(emb1 - emb2, dim=1).item()
        
        distances.append(distance)
        labels.append(1 if is_same.item() else 0)  # 1 for same, 0 for different
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, [-d for d in distances])  # Negative because smaller distance = more similar
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Convert back to positive distance
    optimal_threshold = -optimal_threshold
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', 
                label=f'Optimal threshold: {optimal_threshold:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_analysis.png')
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return optimal_threshold, roc_auc

def compute_accuracy(model_path, test_csv_path, threshold=0.5, num_pairs=2000):
    """
    Compute accuracy on test set using the specified threshold
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = MaskedFaceDetector(model_path=model_path, device=device)
    detector.siamese_net.eval()
    
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    
    # Generate test pairs
    test_pairs = generate_test_pairs(test_df, num_pairs=num_pairs)
    
    # Create dataset and dataloader
    test_dataset = TestPairDataset(test_pairs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    distances = []
    
    print(f"Evaluating pairs with threshold = {threshold}...")
    for img1_path, img2_path, is_same in tqdm(test_loader):
        # Load and preprocess images
        img1 = cv2.imread(img1_path[0])
        img2 = cv2.imread(img2_path[0])
        
        if img1 is None or img2 is None:
            print(f"Error loading images: {img1_path[0]} or {img2_path[0]}")
            continue
        
        # Predict
        pred_same, distance = detector.predict(img1, img2, threshold=threshold)
        
        y_true.append(is_same.item())
        y_pred.append(pred_same)
        distances.append(distance)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n=== Test Results ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot distance distribution
    same_distances = [d for d, same in zip(distances, y_true) if same]
    diff_distances = [d for d, same in zip(distances, y_true) if not same]
    
    plt.figure(figsize=(10, 6))
    plt.hist(same_distances, bins=30, alpha=0.5, label='Same Person', color='green')
    plt.hist(diff_distances, bins=30, alpha=0.5, label='Different People', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution for Same vs Different People')
    plt.legend()
    plt.savefig('test_distance_distribution.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'distances': distances,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    # Configuration
    MODEL_SAVE_PATH = 'model_full_dataset_pytorch.pth'
    TEST_CSV_PATH = 'test_data_full.csv'
    
    # Find optimal threshold
    print("Finding optimal threshold...")
    optimal_threshold, roc_auc = find_optimal_threshold(MODEL_SAVE_PATH, TEST_CSV_PATH, num_pairs=500)
    
    # Compute accuracy with optimal threshold
    print("\nComputing accuracy with optimal threshold...")
    optimal_results = compute_accuracy(
        MODEL_SAVE_PATH, 
        TEST_CSV_PATH, 
        threshold=optimal_threshold,
        num_pairs=2000
    )
    
    # Also compute with default threshold for comparison
    print("\nComputing accuracy with default threshold (0.5)...")
    default_results = compute_accuracy(
        MODEL_SAVE_PATH, 
        TEST_CSV_PATH, 
        threshold=0.5,
        num_pairs=2000
    )
    
    # Print comparison
    print("\n=== Comparison ===")
    print(f"Optimal threshold ({optimal_threshold:.4f}):")
    print(f"  Accuracy: {optimal_results['accuracy']:.4f}")
    print(f"  F1 Score: {optimal_results['f1']:.4f}")
    print(f"\nDefault threshold (0.5):")
    print(f"  Accuracy: {default_results['accuracy']:.4f}")
    print(f"  F1 Score: {default_results['f1']:.4f}")
    
    # Plot error analysis
    plt.figure(figsize=(12, 5))
    
    # Plot errors with optimal threshold
    plt.subplot(1, 2, 1)
    # False positives
    fp = [(i, d) for i, (d, true, pred) in enumerate(zip(optimal_results['distances'], optimal_results['y_true'], optimal_results['y_pred'])) 
          if not true and pred]
    # False negatives
    fn = [(i, d) for i, (d, true, pred) in enumerate(zip(optimal_results['distances'], optimal_results['y_true'], optimal_results['y_pred'])) 
          if true and not pred]
    
    plt.scatter([i for i, _ in fp], [d for _, d in fp], color='red', marker='x', label='False Positives')
    plt.scatter([i for i, _ in fn], [d for _, d in fn], color='blue', marker='o', label='False Negatives')
    plt.axhline(y=optimal_threshold, color='black', linestyle='--', label=f'Threshold={optimal_threshold:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.title(f'Error Analysis (Optimal Threshold={optimal_threshold:.4f})')
    plt.legend()
    
    # Plot errors with default threshold
    plt.subplot(1, 2, 2)
    # False positives
    fp = [(i, d) for i, (d, true, pred) in enumerate(zip(default_results['distances'], default_results['y_true'], default_results['y_pred'])) 
          if not true and pred]
    # False negatives
    fn = [(i, d) for i, (d, true, pred) in enumerate(zip(default_results['distances'], default_results['y_true'], default_results['y_pred'])) 
          if true and not pred]
    
    plt.scatter([i for i, _ in fp], [d for _, d in fp], color='red', marker='x', label='False Positives')
    plt.scatter([i for i, _ in fn], [d for _, d in fn], color='blue', marker='o', label='False Negatives')
    plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold=0.5')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.title('Error Analysis (Default Threshold=0.5)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('error_analysis.png')
    
if __name__ == "__main__":
    main()