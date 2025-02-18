from changesam.modeling.change_decoder_post_df import ChangeDecoderPostDF
from changesam.modeling.change_decoder_pre_df import ChangeDecoderPreDF
from segment_anything.modeling.transformer import TwoWayTransformer
import torch

embeddings_a = torch.randn((1, 256, 64, 64))
embeddings_b = torch.randn((1, 256, 64, 64))
image_pe = torch.randn((1, 256, 64, 64))
sparse_prompt_embeddings = torch.randn((1, 8, 256))
dense_prompt_embedding = torch.randn((1, 256, 64, 64))

transformer = TwoWayTransformer(8, 256, 16, 2048)
decoder_post_df = ChangeDecoderPostDF(transformer_dim=256, transformer=transformer)
decoder_pre_df = ChangeDecoderPreDF(transformer_dim=256, transformer=transformer)

out_post = decoder_post_df(embeddings_a, embeddings_b, image_pe, sparse_prompt_embeddings, dense_prompt_embedding)
out_pre = decoder_pre_df(embeddings_a, embeddings_b, image_pe, sparse_prompt_embeddings, dense_prompt_embedding)

pass