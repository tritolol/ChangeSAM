from changesam.build_changesam import build_changesam
import torch
import hashlib
#from changesam.modeling.change_decoder_pre_df import ChangeDecoderPreDF
#from changesam.modeling.change_decoder_post_df import ChangeDecoderPostDF

from changesam.modeling.change_decoders import ChangeDecoderPreDF, ChangeDecoderPostDF

def compute_model_hash(model):
    """
    Compute an MD5 hash of all parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        
    Returns:
        str: The resulting MD5 hash as a hexadecimal string.
    """
    md5 = hashlib.md5()
    # Iterate over each parameter in the model.
    for param in model.parameters():
        # Ensure we're working with CPU tensors and detached from the computation graph.
        param_data = param.detach().cpu().numpy()
        # Update the MD5 hash with the bytes of the parameter data.
        md5.update(param_data.tobytes())
    return md5.hexdigest()

for decoder_type in ["predf", "postdf"]:
    # Build the ChangeSam model.
    model = build_changesam(
        sam_checkpoint="/root/mobile_sam.pt",
        encoder_type="mobile_sam_vit_t",
        decoder_type=decoder_type,
        lora_layers=[-2],
        lora_r=4,
        lora_alpha=1
    ).eval()

    embeddings_a = torch.ones((1, 256, 64, 64))
    embeddings_b = torch.ones((1, 256, 64, 64))
    image_pe = torch.ones((1, 256, 64, 64))
    sparse_prompt_embeddings = torch.ones((1, 5, 256))
    dense_prompt_embedding = torch.ones((1, 256, 64, 64))

    with torch.no_grad():
        for p in model.mask_decoder.fusion_layer.parameters():
            p.fill_(1.0)

        for k, v in model.image_encoder.state_dict().items():
            if "w_" in k:
                v.fill_(1.0)

        model.sparse_prompt_embeddings.weight.fill_(1.0)

        #out_pre = model.mask_decoder(embeddings_a, embeddings_b, image_pe, sparse_prompt_embeddings, dense_prompt_embedding)

        #test_emb = model.image_encoder(torch.ones((1, 3, 1024, 1024)))

        out_all = model.forward_with_images({"images_a": torch.ones((1, 3, 1024, 1024)), "images_b": -torch.ones((1, 3, 1024, 1024))})


    if type(model.mask_decoder) == ChangeDecoderPostDF:
        assert torch.allclose(out_all.sum(), torch.tensor(82104392.))
    elif type(model.mask_decoder) == ChangeDecoderPreDF:
        assert torch.allclose(out_all.sum(), torch.tensor(-1767644.5000))

print(compute_model_hash(model.mask_decoder))

pass