# improved_train_snlf.py - Enhanced PyTorch Training Script
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

# Import our improved PyTorch modules
from improved_snlf_model import MaskedFaceDetector, SiameseNetwork, UniformLBPProcessor

class TripletLoss(nn.Module):
    """
    Triplet loss function as described in the paper's Section II-C
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Compute distances as described in the paper
        # d(A,P) = ||f(A) - f(P)||²
        positive_distance = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        
        # d(A,N) = ||f(A) - f(N)||²
        negative_distance = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        
        # L(A,P,N) = max(d(A,P) - d(A,N) + margin, 0)
        losses = torch.relu(positive_distance - negative_distance + self.margin)
        
        # Return mean of all triplet losses
        return torch.mean(losses)

class ImprovedTripletDataset(torch.utils.data.Dataset):
    """
    Enhanced dataset for triplet generation with better sampling strategy
    """
    def __init__(self, dataframe, image_size=(32, 32), lbp_processor=None):
        self.dataframe = dataframe
        self.image_size = image_size
        self.lbp_processor = lbp_processor
        
        # Group by identity as in your original code
        self.identity_groups = dataframe.groupby('identity_label')
        self.identity_labels = list(self.identity_groups.groups.keys())
        
        # Create lookup for masked/unmasked images
        self.masked_lookup = {}
        self.unmasked_lookup = {}
        
        for idx, identity in enumerate(self.identity_labels):
            group = self.identity_groups.get_group(identity)
            
            # Get masked and unmasked images for this identity
            masked = group[group['is_masked'] == True]
            unmasked = group[group['is_masked'] == False]
            
            self.masked_lookup[identity] = masked['image_path'].tolist()
            self.unmasked_lookup[identity] = unmasked['image_path'].tolist()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Sample anchor identity from available identities
        anchor_identity = np.random.choice(self.identity_labels)
        
        # Implement diverse sampling strategy:
        # 25% chance: anchor=unmasked, positive=unmasked, negative=unmasked
        # 25% chance: anchor=masked, positive=masked, negative=masked
        # 25% chance: anchor=unmasked, positive=masked, negative=unmasked
        # 25% chance: anchor=masked, positive=unmasked, negative=masked
        sampling_type = np.random.randint(0, 4)
        
        # Get anchor image
        if sampling_type in [0, 2]:  # Unmasked anchor
            if len(self.unmasked_lookup[anchor_identity]) > 0:
                anchor_path = np.random.choice(self.unmasked_lookup[anchor_identity])
            else:
                # Fallback if no unmasked images
                anchor_path = np.random.choice(self.masked_lookup[anchor_identity])
        else:  # Masked anchor
            if len(self.masked_lookup[anchor_identity]) > 0:
                anchor_path = np.random.choice(self.masked_lookup[anchor_identity])
            else:
                # Fallback if no masked images
                anchor_path = np.random.choice(self.unmasked_lookup[anchor_identity])
        
        # Get positive image (same identity as anchor)
        if sampling_type in [0, 3]:  # Unmasked positive
            if len(self.unmasked_lookup[anchor_identity]) > 1 or (len(self.unmasked_lookup[anchor_identity]) > 0 and sampling_type == 3):
                positive_candidates = [p for p in self.unmasked_lookup[anchor_identity] if p != anchor_path]
                if not positive_candidates:
                    # Fallback
                    positive_candidates = self.masked_lookup[anchor_identity]
                positive_path = np.random.choice(positive_candidates)
            else:
                # Fallback
                positive_candidates = [p for p in self.masked_lookup[anchor_identity] if p != anchor_path]
                if not positive_candidates:
                    # Use same image if necessary
                    positive_path = anchor_path
                else:
                    positive_path = np.random.choice(positive_candidates)
        else:  # Masked positive
            if len(self.masked_lookup[anchor_identity]) > 1 or (len(self.masked_lookup[anchor_identity]) > 0 and sampling_type == 2):
                positive_candidates = [p for p in self.masked_lookup[anchor_identity] if p != anchor_path]
                if not positive_candidates:
                    # Fallback
                    positive_candidates = self.unmasked_lookup[anchor_identity]
                positive_path = np.random.choice(positive_candidates)
            else:
                # Fallback
                positive_candidates = [p for p in self.unmasked_lookup[anchor_identity] if p != anchor_path]
                if not positive_candidates:
                    # Use same image if necessary
                    positive_path = anchor_path
                else:
                    positive_path = np.random.choice(positive_candidates)
        
        # Sample negative (different identity)
        negative_identity = np.random.choice([i for i in self.identity_labels if i != anchor_identity])
        
        # Get negative image based on sampling type
        if sampling_type in [0, 2]:  # Unmasked negative
            if len(self.unmasked_lookup[negative_identity]) > 0:
                negative_path = np.random.choice(self.unmasked_lookup[negative_identity])
            else:
                # Fallback
                negative_path = np.random.choice(self.masked_lookup[negative_identity])
        else:  # Masked negative
            if len(self.masked_lookup[negative_identity]) > 0:
                negative_path = np.random.choice(self.masked_lookup[negative_identity])
            else:
                # Fallback
                negative_path = np.random.choice(self.unmasked_lookup[negative_identity])
        
        # Load and preprocess all images
        anchor_img = self._load_and_preprocess(anchor_path)
        positive_img = self._load_and_preprocess(positive_path)
        negative_img = self._load_and_preprocess(negative_path)
        
        # Return processed triplet
        return anchor_img, positive_img, negative_img
    
    def _load_and_preprocess(self, image_path):
        """Load and preprocess image with LBP if processor provided"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to 32x32 as specified in the paper
        img = cv2.resize(img, self.image_size)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply LBP processing if processor is provided
        if self.lbp_processor:
            img = self.lbp_processor.process_image(img)
        
        # Convert to tensor (normalized)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        return img_tensor

class ImprovedTrainer:
    """
    Enhanced trainer for SN-LF implementation with learning rate cycling
    """
    def __init__(self, model_save_path, epochs=100):
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.detector = MaskedFaceDetector(device=self.device)
        
        # Triplet loss with margin as per paper
        self.criterion = TripletLoss(margin=0.2)
        
        # Training configuration based on paper recommendations
        self.batch_size = 32
        self.epochs = epochs
        self.image_size = (32, 32)  # 32x32 as mentioned in the paper
        self.history_plot_path = 'training_history_improved.png'
        
        # Learning rate bounds based on Section II-B of the paper
        self.min_lr = 0.001  # MinLR
        self.max_lr = 0.005  # MaxLR
    
    def train(self, train_data_path, val_data_path):
        """Train the model with learning rate cycling as per the paper"""
        # Load datasets
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        
        print(f"Training on {len(train_data)} images")
        print(f"Validating on {len(val_data)} images")
        
        # Create improved datasets and dataloaders
        train_dataset = ImprovedTripletDataset(
            train_data, 
            image_size=self.image_size,
            lbp_processor=self.detector.lbp_processor
        )
        val_dataset = ImprovedTripletDataset(
            val_data, 
            image_size=self.image_size,
            lbp_processor=self.detector.lbp_processor
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Optimizer with learning rate scheduler as per paper Section II-B
        optimizer = optim.Adam(self.detector.siamese_net.parameters(), lr=self.min_lr)
        
        # Learning rate cycling scheduler
        # As per paper: cycle every 5 iterations between MinLR and MaxLR
        step_size = 5 * len(train_loader)  # 5 epochs per cycle
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.min_lr,
            max_lr=self.max_lr,
            step_size_up=step_size,
            mode='triangular',
            cycle_momentum=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_epoch = 0
        epochs_without_improvement = 0
        
        print("\nStarting training...")
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training phase
            self.detector.siamese_net.train()
            train_loss = 0.0
            
            for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
                # Move to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                anchor_emb = self.detector.siamese_net(anchor)
                positive_emb = self.detector.siamese_net(positive)
                negative_emb = self.detector.siamese_net(negative)
                
                # Calculate loss
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                train_loss += loss.item()
                
                # Log current learning rate
                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    history['lr'].append(current_lr)
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.detector.siamese_net.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, (anchor, positive, negative) in enumerate(val_loader):
                    # Move to device
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)
                    
                    # Forward pass
                    anchor_emb = self.detector.siamese_net(anchor)
                    positive_emb = self.detector.siamese_net(positive)
                    negative_emb = self.detector.siamese_net(negative)
                    
                    # Calculate loss
                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(self.detector.siamese_net.state_dict(), self.model_save_path)
                print(f"New best model saved at epoch {epoch+1}!")
            else:
                epochs_without_improvement += 1
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.epochs} - {epoch_time:.1f}s - "
                  f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Early stopping (15 epochs without improvement)
            if epochs_without_improvement >= 15:
                print(f"Early stopping triggered! No improvement for {epochs_without_improvement} epochs.")
                break
        
        # Plot training history
        self._plot_training_history(history, best_epoch)
        
        print(f"\nTraining completed! Best model saved at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
        
        return history
    
    def _plot_training_history(self, history, best_epoch):
        """Plot training history with learning rate"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot training and validation loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.axvline(x=best_epoch-1, color='r', linestyle='--', 
                   label=f'Best Model (Epoch {best_epoch})')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        if len(history['lr']) > 0:
            # Resample learning rate to match epochs
            lr_epochs = np.linspace(0, self.epochs-1, len(history['lr']))
            ax2.plot(lr_epochs, history['lr'], label='Learning Rate', color='g')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.history_plot_path)
        plt.close()

# Main execution
if __name__ == "__main__":
    # Configuration
    MODEL_SAVE_PATH = 'improved_snlf_model.pth'
    TRAIN_DATA_PATH = 'train_data_full.csv'
    VAL_DATA_PATH = 'val_data_full.csv'
    
    # Initialize trainer with 100 epochs
    trainer = ImprovedTrainer(MODEL_SAVE_PATH, epochs=100)
    
    # Train the model
    history = trainer.train(TRAIN_DATA_PATH, VAL_DATA_PATH)