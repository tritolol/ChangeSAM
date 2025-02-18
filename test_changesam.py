from changesam.build_changesam import build_changesam_predf_from_sam_vit_h_checkpoint, build_changesam_postdf_from_sam_vit_h_checkpoint
import torch

change_sam_pre = build_changesam_predf_from_sam_vit_h_checkpoint("sam_vit_h_4b8939.pth")
#change_sam_post = build_changesam_postdf_from_sam_vit_h_checkpoint("sam_vit_h_4b8939.pth")

mock_data = {}

embedding = torch.randn((1, 256, 64, 64))

mock_data["embeddings_a"] = embedding
mock_data["embeddings_b"] = embedding


mask = change_sam_pre(mock_data)
#mask = change_sam_post(mock_data)


pass