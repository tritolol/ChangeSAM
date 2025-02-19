from changesam.build_changesam import build_changesam
import torch
import torch.nn.functional as F

def test_forward_with_embeddings():
    """
    Test that a ChangeSAM model built with a given SAM checkpoint can perform a forward pass using image embeddings.
    """
    # Build a ChangeSAM model (using the mobile SAM-ViT-T version and pre-DF decoder)
    model = build_changesam(
        sam_checkpoint="/root/mobile_sam.pt",
        encoder_type="mobile_sam_vit_t",
        decoder_type="predf",
        lora_layers=[-1],
        lora_r=4
    )
    
    # Create a dummy embedding tensor (batch_size=1, channels=256, H=64, W=64)
    embedding = torch.randn((1, 256, 64, 64))
    mock_data = {"embeddings_a": embedding, "embeddings_b": embedding}
    
    # Forward pass using embeddings
    mask = model(mock_data)
    assert mask is not None, "Forward pass with embeddings did not return a result."
    # Optionally, check the shape of the mask
    print("Test forward with embeddings passed. Mask shape:", mask.shape)

def test_forward_with_images():
    """
    Test that a ChangeSAM model can process raw images via forward_with_images.
    """
    model = build_changesam(
        sam_checkpoint="/root/mobile_sam.pt",
        encoder_type="mobile_sam_vit_t",
        decoder_type="predf",
        lora_layers=[-1],
        lora_r=4
    )
    
    # Create dummy image data (batch_size=1, channels=3, H=1024, W=1024)
    image = torch.randn((1, 3, 1024, 1024))
    mock_data = {"images_a": image, "images_b": image}
    
    mask = model.forward_with_images(mock_data)
    assert mask is not None, "Forward pass with images did not return a result."
    print("Test forward with images passed. Mask shape:", mask.shape)

def test_freeze_except_and_checkpoint():
    """
    Test the freeze_except, save_adapted_checkpoint, and load_adapted_checkpoint methods.
    """
    # Build a model with LoRA applied to (for example) the last block
    model = build_changesam(
        sam_checkpoint="/root/mobile_sam.pt",
        encoder_type="mobile_sam_vit_t",
        decoder_type="predf",
        lora_layers=[-1],
        lora_r=4
    )
    
    # Specify that only parameters with "lora_A" or "lora_B" in their names should remain trainable.
    adapt_names = ["lora_A", "lora_B"]
    model.freeze_except(adapt_names)
    
    # Verify that the intended parameters remain trainable while all others are frozen.
    for name, param in model.named_parameters():
        if any(adapt_str in name for adapt_str in adapt_names):
            assert param.requires_grad, f"Parameter {name} should be trainable but is frozen."
        else:
            assert not param.requires_grad, f"Parameter {name} should be frozen but is trainable."
    print("Test freeze_except passed.")
    
    # Save a checkpoint of the adapted parameters.
    checkpoint = model.save_adapted_checkpoint()
    # Check that only keys corresponding to adapted parameters are saved.
    for key in checkpoint.keys():
        assert any(adapt_str in key for adapt_str in adapt_names), f"Unexpected parameter {key} saved in checkpoint."
    print("Test save_adapted_checkpoint passed.")
    
    # Modify the adapted parameters to known values (e.g. ones) for testing.
    for name, param in model.named_parameters():
        if any(adapt_str in name for adapt_str in adapt_names):
            param.data.fill_(1.0)
    
    # Save the new checkpoint.
    new_checkpoint = model.save_adapted_checkpoint()
    
    # Zero out the adapted parameters.
    for name, param in model.named_parameters():
        if any(adapt_str in name for adapt_str in adapt_names):
            param.data.zero_()
    
    # Load the checkpoint.
    model.load_adapted_checkpoint(new_checkpoint, strict=True)
    
    # Verify that adapted parameters have been restored to ones.
    for name, param in model.named_parameters():
        if any(adapt_str in name for adapt_str in adapt_names):
            # We expect that all entries are ones.
            if not torch.allclose(param.data, torch.ones_like(param.data)):
                raise AssertionError(f"Parameter {name} was not loaded correctly.")
    print("Test load_adapted_checkpoint passed.")

if __name__ == "__main__":
    # Run all tests
    test_forward_with_embeddings()
    test_forward_with_images()
    test_freeze_except_and_checkpoint()
