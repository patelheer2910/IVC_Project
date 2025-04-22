# snlf_model_pytorch.py - PyTorch implementation of SN-LF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from skimage.feature import local_binary_pattern

class LBPProcessor:
    """Local Binary Pattern processor for image preprocessing"""
    def __init__(self, radius=1, n_points=8):
        self.radius = radius
        self.n_points = n_points
    
    def process_image(self, image):
        """Process image using uniform LBP algorithm"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply LBP
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')
        
        # Histogram equalization
        lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        equalized = cv2.equalizeHist(lbp_normalized)
        
        return equalized
    
    def preprocess_batch(self, images):
        """Preprocess a batch of images"""
        return np.array([self.process_image(img) for img in images])

class FrequencyFeatureExtractor(nn.Module):
    """Frequency feature perception layer"""
    def __init__(self, alpha=0.5):
        super(FrequencyFeatureExtractor, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        # Split features into high and low frequency
        channels = x.size(1)
        c_high = int(channels * (1 - self.alpha))
        c_low = channels - c_high
        
        # Actually split the channels
        high_freq = x[:, :c_high, :, :]  # First c_high channels (128)
        low_freq = x[:, c_high:, :, :]   # Remaining channels (128)
        
        # Only downsample if feature map is larger than 1x1
        if low_freq.size(2) > 1 and low_freq.size(3) > 1:
            low_freq_downsampled = F.avg_pool2d(low_freq, kernel_size=2, stride=2)
        else:
            low_freq_downsampled = low_freq
        
        # Information exchange between high and low frequency
        if high_freq.size(2) != low_freq_downsampled.size(2) or high_freq.size(3) != low_freq_downsampled.size(3):
            low_freq_upsampled = F.interpolate(low_freq_downsampled, size=high_freq.shape[2:], mode='bilinear', align_corners=False)
        else:
            low_freq_upsampled = low_freq_downsampled
        
        # Exchange information (concatenate original with 10% of the other)
        high_freq_enhanced = torch.cat([high_freq, low_freq_upsampled * 0.1], dim=1)
        low_freq_enhanced = torch.cat([low_freq_downsampled, high_freq * 0.1], dim=1)
        
        return high_freq_enhanced, low_freq_enhanced

class SiameseNetwork(nn.Module):
    """Siamese Network for masked face detection"""
    def __init__(self, input_channels=1):
        super(SiameseNetwork, self).__init__()
        
        # Shared convolutional layers (VGG-like)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        
        # Frequency feature extractor
        self.freq_extractor = FrequencyFeatureExtractor()
        
        # Feature processor for high frequency
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Based on the debug output, FrequencyFeatureExtractor isn't actually reducing the channels
        # It still outputs 256 channels for both high and low freq
        self.high_feat_linear = nn.Linear(256, 256)
        self.low_feat_dense = nn.Linear(256, 256)
        
        # Final embedding layer
        self.embedding_layer = nn.Linear(512, 128)
    
    def forward(self, x):
        # Process through convolutional layers
        x = self.conv_layers(x)
        
        # Extract frequency features
        high_freq, low_freq = self.freq_extractor(x)
        
        # Process high frequency features
        high_feat = self.feature_processor(high_freq)
        high_feat = F.relu(self.high_feat_linear(high_feat))
        
        # Process low frequency features
        low_flat = low_freq.view(low_freq.size(0), -1)
        low_feat = F.relu(self.low_feat_dense(low_flat))
        
        # Combine features
        combined = torch.cat([high_feat, low_feat], dim=1)
        
        # Final embedding
        embedding = self.embedding_layer(combined)
        
        return embedding

class MaskedFaceDetector:
    """Complete system for masked face detection using SN-LF"""
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lbp_processor = LBPProcessor()
        self.siamese_net = SiameseNetwork().to(device)
        
        if model_path:
            self.siamese_net.load_state_dict(torch.load(model_path, map_location=device))
    
    def compute_similarity(self, embedding1, embedding2):
        """Compute Euclidean distance between embeddings"""
        distance = torch.norm(embedding1 - embedding2, dim=1)
        return distance
    
    def predict(self, image1, image2, threshold=0.5):
        """Predict if two images contain the same person"""
        self.siamese_net.eval()
        
        # Preprocess images
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
        
        return distance.item() < threshold, distance.item()

# For testing/debugging
if __name__ == "__main__":
    # Test if the model can be instantiated
    detector = MaskedFaceDetector()
    print("Model initialized successfully!")
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    is_same, distance = detector.predict(dummy_image, dummy_image)
    print(f"Test prediction: Same={is_same}, Distance={distance}")