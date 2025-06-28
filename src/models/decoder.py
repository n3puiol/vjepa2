import math

import torch
import torch.nn as nn
from transformers import VJEPA2Config
from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Layer


# class InverseVJEPA2PatchEmbeddings3D(nn.Module):
#     """
#     Inverts the VJEPA2PatchEmbeddings3D operation.
#     """
#
#     def __init__(
#             self,
#             config: VJEPA2Config,
#             hidden_size: int = 1024,
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.tubelet_size = config.tubelet_size
#
#         self.num_patches_per_dim = config.crop_size // config.patch_size
#
#         self.inv_proj = nn.ConvTranspose3d(
#             in_channels=hidden_size,
#             out_channels=config.in_chans,
#             kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
#             stride=(config.tubelet_size, config.patch_size, config.patch_size),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.transpose(1, 2)
#
#         x = x.view(x.shape[0], self.hidden_size, 8, self.num_patches_per_dim, self.num_patches_per_dim)
#
#         x = self.inv_proj(x)
#         return x.transpose(1, 2)


class VJEPA2Decoder(nn.Module):
    """
    VJEPA2 Decoder - A feedforward network that reconstructs images from encoded representations.
    Parameterized as a ViT-L architecture with output dimension 256 × 256 × 3.
    """

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        # self.config = config
        # self.hidden_size = config.hidden_size
        # self.patch_size = config.patch_size
        self.crop_size = config.crop_size

        # Define transformer layers for processing the encoded tokens
        drop_path_rates = [
            (config.drop_path_rate * i / (config.num_hidden_layers - 1) if config.num_hidden_layers > 1 else 0.0)
            for i in range(config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(
            [
                VJEPA2Layer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for i in range(6)
            ]
        )

        # self.inverse_patch_embed = InverseVJEPA2PatchEmbeddings3D(config)
        # self._load_patch_embed_weights(embeddings)

        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder_pred = nn.Linear(config.hidden_size, config.crop_size * config.crop_size * config.in_chans)
        self.decoder_act = nn.ReLU(inplace=True)

        # self.inv_proj = nn.ConvTranspose3d(
        #     in_channels=16,
        #     out_channels=config.in_chans,
        #     kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
        #     stride=(config.tubelet_size, config.patch_size, config.patch_size),
        # )
        # self.inv_proj_act = nn.ReLU(inplace=True)

    # def _load_patch_embed_weights(self, embeddings: VJEPA2PatchEmbeddings3D):
    #     with torch.no_grad():
    #         self.inverse_patch_embed.inv_proj.weight.data = embeddings.patch_embeddings.proj.weight.data

    def forward(self, encoded_states, head_mask=None):
        hidden_states = encoded_states

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, None, layer_head_mask, False)
            hidden_states = layer_outputs[0]

        x = hidden_states

        x = x.view(x.size(0), 16, 128, x.size(-1))
        x_pooled_per_image = torch.mean(x, dim=2)
        # decoder_input_latent = x_pooled_per_image.view(-1, x_pooled_per_image.size(-1))

        projected_flat_features = self.decoder_pred(x_pooled_per_image)
        projected_flat_features = self.decoder_act(projected_flat_features)

        initial_spatial_features = projected_flat_features.view(
            x.size(0),
            -1,
            3,
            256,
            256
        )

        # decoded_images = self.inv_proj(initial_spatial_features)
        # decoded_images = self.inv_proj_act(decoded_images)
        #
        # final_output = decoded_images.view(x.size(0), 16,
        #                                    decoded_images.size(1), decoded_images.size(2), decoded_images.size(3))

        return initial_spatial_features
