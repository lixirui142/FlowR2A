"""Reward condition encoder for classifier-free guidance training.

`RewardEncoderV7` encodes discrete/continuous/array reward metrics into embeddings
for conditioning the diffusion trajectory decoder."""

import torch
import torch.nn as nn
import math


# -----------------------------------------------------------------------------
# Array Reward Encoder
# -----------------------------------------------------------------------------
class ArrayRewardEncoder(nn.Module):
    """Encode array-like rewards (e.g., ttc_time, ego_areas) to fixed dimension.
    Current implementation: Simple MLP
    """
    def __init__(self, array_size, condition_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(array_size, condition_dim * 2),
            nn.ReLU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, array_size) or (B, *shape) -> flatten
        Returns:
            (B, condition_dim)
        """
        if x.ndim > 2:
            x = x.flatten(start_dim=1)  # (B, array_size)
        return self.encoder(x)  # (B, condition_dim)


# -----------------------------------------------------------------------------
# Sinusoidal positional encoding (for continuous variables EP, PDM Score)
# -----------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    """
    Maps a continuous scalar to a high-dimensional vector, similar to the Time Embedding
    in Transformers. This helps the model capture subtle differences in continuous values
    (e.g., 0.85 vs 0.95).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x shape: [Batch_Size, 1]
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -----------------------------------------------------------------------------
# Reward encoder
# -----------------------------------------------------------------------------
class RewardEncoderV7(nn.Module):
    def __init__(self, embed_dim = 256,
                 condition_dim = 256,
                 noisy_score = True,
                 noise_scale = 0.02,
                 ego_progress_key = "safe_ego_progress_normed",
                 reward_to_drop = None,
                 cond_ttc_time_clip = None):
        """
        Args:
            embed_dim: dimension of the final fused output vector
            condition_dim: embedding dimension for each individual metric

        The reward tokens are always fused into a single [B, embed_dim] vector
        (merge_tokens), and the ttc_time / ego_areas array rewards are always
        encoded (ego_areas downsampled to 8 timesteps).
        """
        super().__init__()

        self.condition_dim = condition_dim
        self.cond_ttc_time_clip = cond_ttc_time_clip

        # Define Reward configuration
        # Format: (Name, Type, Param)
        # Type 'discrete': Param = num_embeddings (2 or 3)
        # Type 'continuous': Param = 1 (input dimension)
        # PDMS and all submetrics by default
        self.reward_specs = [
            ('no_at_fault_collisions', 'discrete', 3),       # 0: NC {0, 0.5, 1}
            ('drivable_area_compliance', 'discrete', 2),     # 1: DAC {0, 1}
            ('driving_direction_compliance', 'discrete', 3), # 2: DDC {0, 0.5, 1}
            ('traffic_light_compliance', 'discrete', 2),     # 3: TLC {0, 1}
            (ego_progress_key, 'continuous', 2),             # 4: EP [0, 1]
            ('time_to_collision_within_bound', 'discrete', 2),# 5: TTC {0, 1}
            ('lane_keeping', 'discrete', 2),                 # 6: LK {0, 1}
            ('history_comfort', 'discrete', 2),              # 7: HC {0, 1}
            ('pdm_score', 'continuous', 1)                   # 8: PDM [0, 1]
        ]

        # Array rewards: ttc_time (40) and ego_areas (8 timesteps)
        self.reward_specs.append(('ttc_time', 'array', 40))
        self.reward_specs.append(('ego_areas', 'array', 8 * 2))

        if reward_to_drop is not None:
            self.reward_specs = [rs for rs in self.reward_specs if rs[0] not in reward_to_drop]

        self.pdm_score_key = "pdm_score"
        self.reward_keys = [v[0] for v in self.reward_specs]
        self.num_rewards = len(self.reward_specs)

        # --- 1. Build independent Embedders ---
        # Use ModuleDict, indexed by name, with no interference between them
        self.embedders = nn.ModuleDict()

        for name, r_type, param in self.reward_specs:
            if r_type == 'discrete':
                # Discrete variable: independent Embedding table
                self.embedders[name] = nn.Embedding(param, condition_dim)
            elif r_type == 'continuous':
                # Continuous variable: sinusoidal positional embedding + MLP
                self.embedders[name] = nn.Sequential(
                    SinusoidalPosEmb(condition_dim),
                    nn.Linear(condition_dim, condition_dim * 4),
                    nn.Mish(),
                    nn.Linear(condition_dim * 4, condition_dim),
                )
            elif r_type == 'array':
                # Array variable: use ArrayRewardEncoder
                self.embedders[name] = ArrayRewardEncoder(param, condition_dim)


        # --- 2. Independent Null Embeddings ---
        # Define an independent learnable Null vector for each metric
        self.null_embeddings = nn.Parameter(torch.randn(self.num_rewards, condition_dim) * 0.02)

        # --- 3. Fusion layer: fuse all reward tokens into one vector ---
        total_concat_dim = condition_dim * self.num_rewards
        self.output_mlp = nn.Sequential(
            nn.Linear(total_concat_dim, total_concat_dim),
            nn.SiLU(),
            nn.Linear(total_concat_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.noisy_score = noisy_score
        self.noise_scale = noise_scale



    def _map_ternary_to_index(self, x):
        """Helper function: map {0, 0.5, 1} to indices {0, 1, 2}"""
        return (x * 2).long().clamp(0, 2)

    def forward(self, rewards_dict, drop_mask=None, eval = False):
        """
        rewards_dict: dictionary containing data; Keys must correspond to names in self.reward_specs
        drop_mask: [B, num_rewards] Boolean Tensor. True means drop (use Null).
        """
        # Use any tensor to get batch information
        batch_size = rewards_dict[self.pdm_score_key].shape[0]
        device = rewards_dict[self.pdm_score_key].device

        if drop_mask is None:
            drop_mask = torch.zeros((batch_size, self.num_rewards), dtype=torch.bool, device=device)
        else:
            drop_mask = drop_mask.bool().to(device)

        # --- 1. Iterate over Specs and process each Reward ---
        embs_list = []

        for i, (name, r_type, param) in enumerate(self.reward_specs):
            data = rewards_dict[name]
            encoder = self.embedders[name]

            if r_type == 'discrete':
                # Handle Ternary / Binary
                if encoder.num_embeddings == 3:
                    indices = self._map_ternary_to_index(data)
                else:
                    indices = data.long()

                # [B, dim]
                feat = encoder(indices)

            elif r_type == 'continuous':
                # noise augmentation for continuous reward
                if self.noisy_score and not eval:
                    data = data + torch.randn_like(data) * self.noise_scale
                # [B] -> [B, 1] -> Linear -> [B, dim]
                feat = encoder(data.view(-1, 1))
            elif r_type == 'array':
                # Tackling out of bound ttc_time input.
                if name == 'ttc_time' and self.cond_ttc_time_clip is not None:
                    data = data.clamp(max=self.cond_ttc_time_clip)
                feat = encoder(data.view(-1, param).float())  # (B, dim)


            embs_list.append(feat)

        # --- 2. Stack all rewards ---
        # Shape: [B, num_rewards, dim]
        stack_features = torch.stack(embs_list, dim=1)

        # --- 3. Mask Substitution (Vectorized) ---
        # Expand Null Embeddings: [num_rewards, dim] -> [B, num_rewards, dim]
        null_features = self.null_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Expand Mask: [B, num_rewards] -> [B, num_rewards, 1] -> [B, num_rewards, dim]
        mask_expanded = drop_mask.unsqueeze(-1).expand(-1, -1, self.condition_dim)

        # Apply Mask: True -> Null, False -> Real
        dropd_features = torch.where(mask_expanded, null_features, stack_features)

        # --- 4. Fuse into a single Global Vector ---
        # [B, num_rewards, dim] -> [B, num_rewards*dim]
        flattened_features = dropd_features.reshape(batch_size, -1)
        output = self.output_mlp(flattened_features)

        return output
