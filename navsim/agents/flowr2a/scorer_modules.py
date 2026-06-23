"""
Scorer transformer modules for trajectory quality evaluation.

Contains the ScorerTransformerDecoderLayer and ScorerTransformerDecoder used
by the coarse and fine scoring heads in TrajectoryHead.
"""
import copy
import torch
import torch.nn as nn

from navsim.agents.flowr2a.modules.blocks import GridSampleCrossBEVAttentionScorer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ScorerTransformerDecoderLayer(nn.Module):
    """Single layer of the scorer transformer: BEV cross-attn, agent cross-attn,
    optional self-attn, ego cross-attn, and FFN."""

    def __init__(self,
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 scorer_d_model: int = 512,
                 scorer_d_ffn: int = 2048,
                 scorer_num_head: int = 16,
                 scorer_dropout: float = 0.2):
        super().__init__()

        self.cross_bev_attention = GridSampleCrossBEVAttentionScorer(
            scorer_d_model,
            scorer_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=d_model,
        )
        self.mha_input = nn.Linear(d_model, scorer_d_model)
        self.cross_scene_attention = nn.MultiheadAttention(
            scorer_d_model,
            scorer_num_head,
            dropout=scorer_dropout,
            batch_first=True,
        )
        self.norm_scene = nn.LayerNorm(scorer_d_model)

        self.agent_input = nn.Linear(d_model, scorer_d_model)
        self.cross_agent_attention = nn.MultiheadAttention(
            scorer_d_model,
            scorer_num_head,
            dropout=scorer_dropout,
            batch_first=True,
        )

        self.norm_bev = nn.LayerNorm(scorer_d_model)

        self.ffn = nn.Sequential(
            nn.Linear(scorer_d_model, scorer_d_ffn),
            nn.ReLU(),
            nn.Linear(scorer_d_ffn, scorer_d_model),
        )
        self.norm1 = nn.LayerNorm(scorer_d_model)
        self.norm4 = nn.LayerNorm(scorer_d_model)

    def _bev_cross_attn(self, traj_feature, traj_points, bev_feature, bev_spatial_shape, keyval=None):
        """Pre-norm BEV cross-attention via grid_sample (residual computed by caller)."""
        out = self.cross_bev_attention(self.norm_bev(traj_feature), traj_points, bev_feature, bev_spatial_shape)
        # tmp workaround
        return out - traj_feature

    def forward(self,
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                keyval=None):
        # Pre-norm + residual for grid BEV cross-attention
        traj_feature = traj_feature + self._bev_cross_attn(traj_feature, traj_points, bev_feature, bev_spatial_shape, keyval=keyval)
        # MHA cross-attention to the scene (keyval)
        mha_kv = self.mha_input(keyval)
        traj_feature = traj_feature + self.cross_scene_attention(self.norm_scene(traj_feature), mha_kv, mha_kv)[0]
        # Pre-norm + residual for agent cross-attention (ego + agents concatenated)
        combined_query = torch.cat([ego_query, agents_query], dim=1)
        combined_query = self.agent_input(combined_query)
        traj_feature = traj_feature + self.cross_agent_attention(self.norm1(traj_feature), combined_query, combined_query)[0]
        # Pre-norm + residual for FFN
        traj_feature = traj_feature + self.ffn(self.norm4(traj_feature))

        return traj_feature


class ScorerTransformerDecoder(nn.Module):
    """Multi-layer scorer decoder that returns intermediate features from each layer."""

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self,
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                keyval=None):
        traj_feature_list = []
        for mod in self.layers:
            traj_feature = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, keyval=keyval)
            traj_feature_list.append(traj_feature)
        return traj_feature_list
