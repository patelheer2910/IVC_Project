# train_snlf_pytorch.py - PyTorch Training Script for SN-LF
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Import our PyTorch modules
from snlf_model_pytorch import MaskedFaceDetector
from triplet_dataset_pytorch import FERETTripletDataset

class TripletLoss(nn.Module):
    """Triplet loss function"""
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        positive_distance = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        negative_distance = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        
        losses = torch.relu(positive_distance - negative_distance + self.margin)
        return torch.mean(losses)

class OptimizedTrainer:
    """
    Optimized trainer for 700 images, 1 GPU, 50 epochs using PyTorch
    """
    def __init__(self, model_save_path):
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.detector = MaskedFaceDetector(device=self.device)
        self.criterion = TripletLoss()
        
        # Training configuration
        self.batch_size = 32
        self.epochs = 50
        self.image_size = (32, 32)
    
    def train(self, train_data_path, val_data_path):
        """Train the model"""
        # Load datasets
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        
        # Create datasets and dataloaders
        train_dataset = FERETTripletDataset(
            train_data, 
            image_size=self.image_size,
            lbp_processor=self.detector.lbp_processor
        )
        val_dataset = FERETTripletDataset(
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
        
        # Optimizer with learning rate scheduler
        optimizer = optim.Adam(self.detector.siamese_net.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
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
                
                train_loss += loss.item()
            
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
                torch.save(self.detector.siamese_net.state_dict(), self.model_save_path)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.epochs} - {epoch_time:.1f}s - "
                  f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Update learning rate
            scheduler.step()
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History (700 Images, 50 Epochs)')
        plt.savefig('training_history_pytorch.png')
        plt.show()
        
        return history

# Main execution
if __name__ == "__main__":
    # Configuration
    MODEL_SAVE_PATH = 'model_700_images_pytorch.pth'
    TRAIN_DATA_PATH = 'train_data_700.csv'
    VAL_DATA_PATH = 'val_data_700.csv'
    
    # Initialize trainer
    trainer = OptimizedTrainer(MODEL_SAVE_PATH)
    
    # Train the model
    history = trainer.train(TRAIN_DATA_PATH, VAL_DATA_PATH)
    
    print("\nTraining completed!")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    
    # Print final results
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")