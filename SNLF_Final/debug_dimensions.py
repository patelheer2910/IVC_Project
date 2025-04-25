# debug_dimensions.py - Script to debug network dimensions
import torch
import numpy as np
import cv2
from improved_snlf_model import SiameseNetwork, FrequencyFeaturePerception, UniformLBPProcessor

def debug_network_dimensions():
    """Test network to determine exact dimensions at each step"""
    print("=== Starting Network Dimension Debugging ===")
    
    # Create a network instance
    model = SiameseNetwork()
    
    # Create a dummy input
    batch_size = 32
    channels = 1
    height = 32
    width = 32
    dummy_input = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {dummy_input.shape}")
    
    # Run forward pass step by step with dimension checking
    with torch.no_grad():
        # Step 1: Convolutional layers
        print("\n=== Step 1: Convolutional Layers ===")
        x = model.conv_layers(dummy_input)
        print(f"After conv layers: {x.shape}")
        
        # Step 2: Frequency Feature Extraction
        print("\n=== Step 2: Frequency Feature Extraction ===")
        freq_extractor = FrequencyFeaturePerception()
        high_freq, low_freq = freq_extractor(x)
        print(f"High freq shape: {high_freq.shape}")
        print(f"Low freq shape: {low_freq.shape}")
        
        # Get dimensions for frequency splitting
        print("\n=== Step 3: Feature Splitting ===")
        high_channels = high_freq.size(1)
        low_channels = low_freq.size(1)
        print(f"High freq channels: {high_channels}")
        print(f"Low freq channels: {low_channels}")
        
        # Debug the 4-group division
        print("\n=== Step 4: Debug 4-group Division ===")
        # Split high freq into width and height groups
        high_width = high_freq[:, :high_channels//2, :, :]
        high_height = high_freq[:, high_channels//2:, :, :]
        print(f"High width shape: {high_width.shape}")
        print(f"High height shape: {high_height.shape}")
        
        # Split low freq into width and height groups
        low_width = low_freq[:, :low_channels//2, :, :]
        low_height = low_freq[:, low_channels//2:, :, :]
        print(f"Low width shape: {low_width.shape}")
        print(f"Low height shape: {low_height.shape}")
        
        # Debug the concatenation operations
        print("\n=== Step 5: Debug Concatenation ===")
        print(f"Concatenating: high_width {high_width.shape} with low_width {low_width.shape}")
        
        # Check spatial dimensions
        print(f"High width spatial dims: {high_width.size(2)}x{high_width.size(3)}")
        print(f"Low width spatial dims: {low_width.size(2)}x{low_width.size(3)}")
        
        # The issue is likely here - check if spatial dimensions match
        if high_width.size(2) != low_width.size(2) or high_width.size(3) != low_width.size(3):
            print("ERROR: Spatial dimensions don't match for concatenation!")
            
            # Try fixing with interpolation
            print("Attempting to fix with interpolation...")
            low_width_resized = torch.nn.functional.interpolate(
                low_width,
                size=(high_width.size(2), high_width.size(3)),
                mode='bilinear',
                align_corners=False
            )
            print(f"Resized low width shape: {low_width_resized.shape}")
            
            # Try concatenation with fixed dimensions
            try:
                hw_lw_fixed = torch.cat([high_width, low_width_resized], dim=1)
                print(f"Fixed concatenation shape: {hw_lw_fixed.shape}")
            except Exception as e:
                print(f"Fixed concatenation still failed: {e}")
        else:
            # Try original concatenation
            try:
                hw_lw = torch.cat([high_width, low_width], dim=1)
                print(f"Concatenation shape: {hw_lw.shape}")
            except Exception as e:
                print(f"Concatenation failed: {e}")
        
        # Debug all four concatenations
        print("\n=== Step 6: Debug All Four Concatenations ===")
        
        # Function to safely test concatenation with interpolation if needed
        def safe_concat(a, b, name):
            print(f"Testing {name} concatenation: {a.shape} + {b.shape}")
            if a.size(2) != b.size(2) or a.size(3) != b.size(3):
                print(f"  Spatial dimensions don't match, resizing...")
                b_resized = torch.nn.functional.interpolate(
                    b,
                    size=(a.size(2), a.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                try:
                    result = torch.cat([a, b_resized], dim=1)
                    print(f"  Success after resize: {result.shape}")
                    return result
                except Exception as e:
                    print(f"  Failed after resize: {e}")
                    return None
            else:
                try:
                    result = torch.cat([a, b], dim=1)
                    print(f"  Success: {result.shape}")
                    return result
                except Exception as e:
                    print(f"  Failed: {e}")
                    return None
        
        # Test all four combinations
        hw_lw_result = safe_concat(high_width, low_width, "hw_lw")
        hw_lh_result = safe_concat(high_width, low_height, "hw_lh")
        hh_lw_result = safe_concat(high_height, low_width, "hh_lw")
        hh_lh_result = safe_concat(high_height, low_height, "hh_lh")
        
        # Debug the convolution layers
        print("\n=== Step 7: Debug Convolution Layers ===")
        if hw_lw_result is not None:
            try:
                hw_lw_conv = model.high_width_low_width(hw_lw_result)
                print(f"hw_lw_conv shape: {hw_lw_conv.shape}")
            except Exception as e:
                print(f"hw_lw_conv failed: {e}")
        
        # Debug feature processing
        print("\n=== Step 8: Debug Feature Processing ===")
        try:
            high_feat = model.high_freq_processor(high_freq)
            print(f"High freq processed shape: {high_feat.shape}")
            
            low_feat = model.low_freq_processor(low_freq)
            print(f"Low freq processed shape: {low_feat.shape}")
            
            # Linear layers
            high_feat_processed = model.high_feat_dense(high_feat)
            print(f"High feat dense shape: {high_feat_processed.shape}")
            
            low_feat_processed = model.low_feat_dense(low_feat)
            print(f"Low feat dense shape: {low_feat_processed.shape}")
            
            # Combined
            combined = torch.cat([high_feat_processed, low_feat_processed], dim=1)
            print(f"Combined features shape: {combined.shape}")
            
            # Final embedding
            embedding = model.embedding_layer(combined)
            print(f"Final embedding shape: {embedding.shape}")
        except Exception as e:
            print(f"Feature processing failed: {e}")
    
    # Try full forward pass with simplified model
    print("\n=== Step 9: Suggested Fix ===")
    print("Based on the diagnostics, the issue is in the 4-group division and concatenation.")
    print("Here's a suggested fix for the forward method:\n")
    print("""
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
    """)
    
    # Test final forward pass with dimension tracking
    print("\n=== Step 10: Full Forward Pass Test ===")
    try:
        # Create a simplified version of the model's forward method for testing
        def simplified_forward(model, x):
            # Process through convolutional layers
            x = model.conv_layers(x)
            
            # Extract frequency features
            high_freq, low_freq = model.freq_extractor(x)
            
            # Process high and low frequency features separately
            high_feat = model.high_freq_processor(high_freq)
            low_feat = model.low_freq_processor(low_freq)
            
            # Apply non-linear transformations
            high_feat = torch.nn.functional.relu(model.high_feat_dense(high_feat))
            low_feat = torch.nn.functional.relu(model.low_feat_dense(low_feat))
            
            # Combine features
            combined = torch.cat([high_feat, low_feat], dim=1)
            
            # Final embedding
            embedding = model.embedding_layer(combined)
            
            return embedding
        
        output = simplified_forward(model, dummy_input)
        print(f"Simplified forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Simplified forward pass failed: {e}")
    
    print("\n=== Debugging Complete ===")
    print("Modify your SiameseNetwork forward method according to the suggested fix.")

if __name__ == "__main__":
    debug_network_dimensions()