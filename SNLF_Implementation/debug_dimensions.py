# debug_dimensions.py - Script to debug network dimensions
import torch
from snlf_model_pytorch import SiameseNetwork, FrequencyFeatureExtractor

def debug_network_dimensions():
    """Test network to determine exact dimensions"""
    # Create a network instance
    model = SiameseNetwork()
    model._debug_print = True  # Enable debug mode
    
    # Create a dummy input
    batch_size = 32
    channels = 1
    height = 32
    width = 32
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Run forward pass to see the dimensions
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")
        
        # Let's check intermediate dimensions manually
        with torch.no_grad():
            x = model.conv_layers(dummy_input)
            print(f"After conv layers: {x.shape}")
            
            # Test FrequencyFeatureExtractor directly
            freq_extractor = FrequencyFeatureExtractor()
            high_freq, low_freq = freq_extractor(x)
            print(f"High freq shape: {high_freq.shape}")
            print(f"Low freq shape: {low_freq.shape}")
            
            # Calculate expected dimensions
            high_channels = high_freq.shape[1]
            low_channels = low_freq.shape[1]
            
            print(f"High freq channels: {high_channels}")
            print(f"Low freq channels: {low_channels}")
            
            # Let's check what happens with feature processor
            high_flat = high_freq.view(high_freq.size(0), -1)
            print(f"High freq flattened shape: {high_flat.shape}")
            
            low_flat = low_freq.view(low_freq.size(0), -1)
            print(f"Low freq flattened shape: {low_flat.shape}")
            
            # Test splitting manually
            channels = x.size(1)
            c_high = int(channels * 0.5)
            c_low = channels - c_high
            print(f"Expected split - high: {c_high}, low: {c_low}")
            
            # Test actual splitting
            high_split = x[:, :c_high, :, :]
            low_split = x[:, c_high:, :, :]
            print(f"Split high shape: {high_split.shape}")
            print(f"Split low shape: {low_split.shape}")

if __name__ == "__main__":
    debug_network_dimensions()