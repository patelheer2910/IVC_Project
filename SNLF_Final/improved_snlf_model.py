# improved_snlf_model.py - PyTorch implementation of SN-LF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from skimage.feature import local_binary_pattern
import math

class UniformLBPProcessor:
    """
    Uniform Local Binary Pattern processor for image preprocessing
    Implementation based on the paper's description of texture feature extraction
    """
    def __init__(self, radius=1, n_points=8, method='uniform'):
        self.radius = radius
        self.n_points = n_points
        self.method = method
        # Number of possible patterns with uniform LBP
        # As per paper: i*(i-1)+3 where i is number of domain pixels
        self.n_patterns = n_points * (n_points - 1) + 3
        
    def process_image(self, image):
        """Process image using uniform LBP algorithm with spatial histogram"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Uniform LBP as specified in the paper
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method=self.method)
        
        # Histogram equalization to standardize contrast
        lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        equalized = cv2.equalizeHist(lbp_normalized)
        
        # Create spatial histogram as mentioned in the paper (Section II-A)
        # Divide face map into blocks to capture spatial structure
        M = 4  # Number of blocks (can be adjusted)
        h, w = equalized.shape
        block_h, block_w = h // 2, w // 2
        
        # This is for feature visualization - we return the original LBP image
        # The actual features will be computed in the forward pass of the network
        return equalized
    
    def compute_spatial_histogram(self, lbp_image):
        """
        Compute spatial histogram features for LBP image
        Divides image into M blocks and computes histogram for each
        """
        M = 4  # Number of blocks (2x2 grid)
        h, w = lbp_image.shape
        block_h, block_w = h // 2, w // 2
        
        # Initialize spatial histogram
        histograms = []
        
        # Process each block
        for i in range(2):
            for j in range(2):
                # Extract block
                block = lbp_image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                
                # Compute histogram for this block
                hist, _ = np.histogram(block, bins=self.n_patterns, range=(0, self.n_patterns-1), density=True)
                histograms.append(hist)
        
        # Concatenate histograms to get spatial histogram for complete face
        spatial_histogram = np.concatenate(histograms)
        
        return spatial_histogram
    
    def preprocess_batch(self, images):
        """Preprocess a batch of images"""
        return np.array([self.process_image(img) for img in images])

class FrequencyFeaturePerception(nn.Module):
    """
    Frequency feature perception layer as described in the paper
    Divides features into high and low frequency components with information exchange
    """
    def __init__(self, alpha=0.5):
        super(FrequencyFeaturePerception, self).__init__()
        self.alpha = alpha  # Controls division between high/low frequency features
        
    def forward(self, x):
        # Get dimensions
        batch_size, channels, height, width = x.size()
        
        # Split channels into high and low frequency features
        # As per paper: alpha controls the ratio
        c_high = int(channels * (1 - self.alpha))
        c_low = channels - c_high
        
        # Split the input tensor along channel dimension
        high_freq = x[:, :c_high, :, :]          # High frequency features
        low_freq = x[:, c_high:, :, :]           # Low frequency features
        
        # Downsample low frequency features to reduce spatial dimensions by half
        # This increases receptive field and reduces computation as mentioned in paper
        low_freq_downsampled = F.avg_pool2d(low_freq, kernel_size=2, stride=2)
        
        # Information exchange between high and low frequency components
        # For proper information exchange, we need to match dimensions
        
        # Upsample low freq features to match high freq spatial dimensions for exchange
        if height > 1 and width > 1:
            low_freq_upsampled = F.interpolate(
                low_freq_downsampled, 
                size=(height, width),
                mode='bilinear', 
                align_corners=False
            )
        else:
            low_freq_upsampled = low_freq_downsampled
            
        # Information exchange as specified in the paper
        # High freq features receive 10% information from low freq
        # Low freq features receive 10% information from high freq
        exchange_ratio = 0.1
        
        # Process high frequency features with information from low frequency
        # Expand dimensions for proper concatenation
        if low_freq_upsampled.size(2) != high_freq.size(2) or low_freq_upsampled.size(3) != high_freq.size(3):
            low_freq_for_high = F.interpolate(
                low_freq_upsampled, 
                size=(high_freq.size(2), high_freq.size(3)),
                mode='bilinear', 
                align_corners=False
            )
        else:
            low_freq_for_high = low_freq_upsampled
        
        # Information exchange
        high_freq_enhanced = torch.cat([
            high_freq, 
            low_freq_for_high * exchange_ratio
        ], dim=1)
        
        # Ensure high_freq has reduced number of output channels
        # by using 1x1 convolution to reduce channels back to original high_freq channels
        high_freq_conv = nn.Conv2d(
            high_freq_enhanced.size(1), c_high, kernel_size=1
        ).to(high_freq.device)
        high_freq_final = high_freq_conv(high_freq_enhanced)
        
        # Process low frequency with information from high frequency
        # Resize high_freq to match low_freq_downsampled for proper exchange
        high_freq_for_low = F.avg_pool2d(high_freq, kernel_size=2, stride=2)
        if high_freq_for_low.size(2) != low_freq_downsampled.size(2) or high_freq_for_low.size(3) != low_freq_downsampled.size(3):
            high_freq_for_low = F.interpolate(
                high_freq_for_low, 
                size=(low_freq_downsampled.size(2), low_freq_downsampled.size(3)),
                mode='bilinear',
                align_corners=False
            )
        
        # Combine for low frequency features
        low_freq_enhanced = torch.cat([
            low_freq_downsampled, 
            high_freq_for_low * exchange_ratio
        ], dim=1)
        
        # Reduce channels back to original low_freq channels
        low_freq_conv = nn.Conv2d(
            low_freq_enhanced.size(1), c_low, kernel_size=1
        ).to(low_freq.device)
        low_freq_final = low_freq_conv(low_freq_enhanced)
        
        return high_freq_final, low_freq_final

class SiameseNetwork(nn.Module):
    """
    Siamese Network for masked face detection with frequency feature perception
    Based on the SN-LF architecture described in the paper
    """
    def __init__(self, input_channels=1):
        super(SiameseNetwork, self).__init__()
        
        # Convolutional layers based on the paper's description
        self.conv_layers = nn.Sequential(
            # First convolutional group
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Second convolutional group
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Third convolutional group
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Fourth convolutional group
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Frequency feature perception layer
        self.freq_extractor = FrequencyFeaturePerception(alpha=0.5)
        
        # Feature processors for high and low frequency
        self.high_freq_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.low_freq_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Feature integration layers - fixed dimensions
        self.high_feat_dense = nn.Linear(128, 128)  # Matches high_freq channels
        self.low_feat_dense = nn.Linear(128, 128)   # Matches low_freq channels
        
        # Final embedding layer (128-dimensional as per paper)
        self.embedding_layer = nn.Linear(256, 128)  # 128 + 128 = 256
    
    def forward(self, x):
        # Process through convolutional layers
        x = self.conv_layers(x)
        
        # Extract frequency features
        high_freq, low_freq = self.freq_extractor(x)
        
        # Process high and low frequency features separately
        high_feat = self.high_freq_processor(high_freq)
        low_feat = self.low_freq_processor(low_freq)
        
        # Apply non-linear transformations
        high_feat = F.relu(self.high_feat_dense(high_feat))
        low_feat = F.relu(self.low_feat_dense(low_feat))
        
        # Combine features
        combined = torch.cat([high_feat, low_feat], dim=1)
        
        # Final embedding
        embedding = self.embedding_layer(combined)
        
        return embedding

class MaskedFaceDetector:
    """
    Complete system for masked face detection using SN-LF as in the paper
    """
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lbp_processor = UniformLBPProcessor()
        self.siamese_net = SiameseNetwork().to(device)
        
        # Load pre-trained model if provided
        if model_path:
            self.siamese_net.load_state_dict(torch.load(model_path, map_location=device))
            self.siamese_net.eval()
    
    def compute_similarity(self, embedding1, embedding2):
        """Compute Euclidean distance between embeddings"""
        # Use L2 distance as per paper's Section II-C
        distance = torch.norm(embedding1 - embedding2, dim=1)
        return distance
    
    def predict(self, image1, image2, threshold=0.5):
        """Predict if two images contain the same person"""
        self.siamese_net.eval()
        
        # Preprocess images with LBP as specified in paper's Section II-A
        img1_lbp = self.lbp_processor.process_image(image1)
        img2_lbp = self.lbp_processor.process_image(image2)
        
        # Convert to tensor and add batch/channel dimensions
        img1_tensor = torch.from_numpy(img1_lbp).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
        img2_tensor = torch.from_numpy(img2_lbp).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
        
        # Get embeddings
        with torch.no_grad():
            emb1 = self.siamese_net(img1_tensor)
            emb2 = self.siamese_net(img2_tensor)
        
        # Compute similarity
        distance = self.compute_similarity(emb1, emb2)
        
        # Use threshold to determine if same person (smaller distance = more similar)
        return distance.item() < threshold, distance.item()