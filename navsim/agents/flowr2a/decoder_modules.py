"""
Diffusion decoder modules for the flow-based trajectory decoder.

Contains the CustomTransformerDecoder (with reward conditioning and CFG support),
its building blocks (AdaLayerNormContinuous, GEGLU, SeqPosEmb), and helpers.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.flowr2a.modules.blocks import (
    gen_sineembed_for_position,
    SinusoidalPosEmb,
)


def _get_clones(module, N):
    """Deep-copy a module N times into a ModuleList."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding` is upcasted to float32
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class GEGLU(nn.Module):
    r"""
    A [variant](https://huggingface.co/papers/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class CustomTransformerDecoderLayer(nn.Module):
    """Single diffusion decoder layer with self-attn, BEV cross-attn, optional agent cross-attn, and FFN.
    All normalizations use AdaLayerNormContinuous conditioned on timestep (+ reward) embedding."""

    def __init__(self,
                 d_model,
                 d_ffn,
                 num_head,
                 dropout,
                 ada_dim=None):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            d_model,
            num_head,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_bev_attention = nn.MultiheadAttention(
            d_model,
            num_head,
            dropout=dropout,
            batch_first=True,
        )

        ada_dim = d_model if ada_dim is None else ada_dim
        self.cross_agent_attention = nn.MultiheadAttention(
            d_model,
            num_head,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            GEGLU(d_model, d_ffn),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = AdaLayerNormContinuous(d_model, ada_dim)
        self.norm2 = AdaLayerNormContinuous(d_model, ada_dim)
        self.norm3 = AdaLayerNormContinuous(d_model, ada_dim)
        self.norm4 = AdaLayerNormContinuous(d_model, ada_dim)

        self.agent_drop_out = nn.Dropout(0.1)

    def forward(self,
                traj_feature,
                bev_feature,
                agents_feature,
                time_embed):
        normed = self.norm1(traj_feature, time_embed)
        traj_feature = traj_feature + self.self_attention(normed, normed, normed)[0]

        normed = self.norm2(traj_feature, time_embed)
        traj_feature = traj_feature + self.cross_bev_attention(normed, bev_feature, bev_feature)[0]

        normed = self.norm3(traj_feature, time_embed)
        traj_feature = traj_feature + self.agent_drop_out(self.cross_agent_attention(normed, agents_feature, agents_feature)[0])

        normed = self.norm4(traj_feature, time_embed)
        traj_feature = traj_feature + self.ffn(normed)

        return traj_feature


class SeqPosEmb(nn.Module):
    """Sinusoidal positional embedding for sequences (pre-computed buffer)."""

    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        """x shape: [1, seq_len] or scalar — returns positional embedding for the sequence length."""
        seq_len = x.shape[1]
        return self.pe[:, :seq_len, :]


class CustomTransformerDecoder(nn.Module):
    """Multi-layer diffusion decoder with reward conditioning injection (adaln/concat/cross)."""

    def __init__(
        self,
        decoder_layer,
        num_layers,
        d_model,
        sin_dim=64,
        action_dim=4,
        horizon=8,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.sin_dim = sin_dim
        self.action_dim = action_dim
        dim_in = sin_dim + 2 # sin_dim for embeded x,y. 2 dim for heading.

        self.proj_in_traj = nn.Linear(dim_in, d_model)
        self.proj_out_traj = nn.Linear(d_model, action_dim)  # to x,y,heading_sin,heading_cos
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        self.traj_pos_emb = SeqPosEmb(d_model, max_len=horizon)

        self.input_norm = nn.LayerNorm(d_model)

    def traj_encoding(self, raw_traj):
        """Encode raw trajectory (x,y,sin,cos) into d_model-dimensional tokens with positional embedding."""
        traj_len = raw_traj.shape[1]
        device = raw_traj.device
        traj_pos_emb = gen_sineembed_for_position(raw_traj, hidden_dim=self.sin_dim)
        traj_emb = torch.cat([traj_pos_emb, raw_traj[..., 2:]], dim=-1)

        traj_emb = self.proj_in_traj(traj_emb)
        pos_ids = torch.arange(traj_len, device=device).unsqueeze(0)
        t_pos_emb = self.traj_pos_emb(pos_ids)
        traj_emb = traj_emb + t_pos_emb

        return traj_emb

    def forward(self,
                sample,
                bev_feature,
                agents_feature,
                timestep,
                reward_cond_emb):
        """
        Forward pass of the diffusion decoder.

        Args:
            sample: noisy trajectory in normalized space, (B, T, 4)
            bev_feature: BEV feature map or flattened tokens
            agents_feature: agent query features
            timestep: diffusion timestep scalar or (B,)
            reward_cond_emb: reward conditioning embedding, (B, D)

        Returns:
            x_pred: predicted trajectory in normalized space, (B, T, 4)
        """
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])

        # scale from [0-1] to [0-1000]
        timesteps = timesteps * 1000
        time_embed = self.time_mlp(timesteps)

        traj_emb = self.traj_encoding(sample)

        input_tokens = traj_emb
        cross_tokens = bev_feature

        # Reward conditioning is always injected via adaln (concat with time embedding)
        cond_embed = torch.cat([time_embed, reward_cond_emb], dim=-1)

        input_tokens = self.input_norm(input_tokens)

        for mod in self.layers:
            input_tokens = mod(input_tokens, cross_tokens, agents_feature, cond_embed)

        x_pred = self.proj_out_traj(input_tokens)

        return x_pred
