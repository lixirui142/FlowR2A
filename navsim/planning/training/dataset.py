from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import logging
import pickle
import gzip
import os

import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
import pdb
logger = logging.getLogger(__name__)
DEBUG = os.environ["DEBUG"] if "DEBUG" in os.environ else "0"


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: Optional[List[str]] = None,
        train = True,
        reward_cache_path: Optional[str] = None,
        valid_cache_paths: Optional[Dict[str, Path]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        :param reward_cache_path: optional separate cache dir for reward caches.
            If None, reward caches are loaded from cache_path (same as feature/target).
        :param valid_cache_paths: optional pre-computed dict of token -> Path.
            If provided, skips the slow _load_valid_caches scan entirely.
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)
        self._reward_cache_path = Path(reward_cache_path) if reward_cache_path is not None else None

        self._feature_builders = feature_builders
        self._target_builders = target_builders

        if valid_cache_paths is not None:
            logger.info("Using pre-loaded valid_cache_paths (%d tokens)", len(valid_cache_paths))
            self._valid_cache_paths = valid_cache_paths
        else:
            if log_names is not None:
                self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
            else:
                self.log_names = [log_name for log_name in self._cache_path.iterdir()]
            self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
                cache_path=self._cache_path,
                feature_builders=self._feature_builders,
                target_builders=self._target_builders,
                log_names=self.log_names,
            )
        self.tokens = list(self._valid_cache_paths.keys())
        self.train = train
        if train:
            # Per-scene reward caches (transfuser_reward.gz) are required for training.
            sample_token_path = next(iter(self._valid_cache_paths.values()), None)
            sample_reward_path = self._get_reward_path(sample_token_path) if sample_token_path else None
            if sample_reward_path and sample_reward_path.exists():
                logger.info("Using per-scene transfuser_reward.gz caches (reward_cache_path=%s)", self._reward_cache_path or self._cache_path)

    def _get_reward_path(self, token_path: Path) -> Path:
        """Get the reward cache path for a given token_path, respecting reward_cache_path override."""
        if self._reward_cache_path is not None:
            rel = token_path.relative_to(self._cache_path)
            return self._reward_cache_path / rel / "transfuser_reward.gz"
        return token_path / "transfuser_reward.gz"

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        # pdb.set_trace()
        return self._load_scene_with_token(self.tokens[idx])

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if DEBUG == "1":
            log_names = log_names[:5]

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)
        
        features["token"] = token

        token_path = self._valid_cache_paths[token]
        features["token_path"] = str(token_path)
        if self.train:
            reward_path = self._get_reward_path(token_path)
            if reward_path.exists():
                features["reward"] = load_feature_target_from_pickle(reward_path)

        return (features, targets)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        train = True,
        reward_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self._scene_loader = scene_loader
        self._feature_builders = feature_builders
        self._target_builders = target_builders

        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None
        self._reward_cache_path = Path(reward_cache_path) if reward_cache_path is not None else None
        self._force_cache_computation = force_cache_computation
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )

        if self._cache_path is not None:
            self.cache_dataset()

        self.train = train
        if train:
            # Per-scene reward caches (transfuser_reward.gz) are required for training.
            sample_token_path = next(iter(self._valid_cache_paths.values()), None)
            sample_reward_path = self._get_reward_path(sample_token_path) if sample_token_path else None
            if sample_reward_path and sample_reward_path.exists():
                logger.info("Using per-scene transfuser_reward.gz caches (reward_cache_path=%s)", self._reward_cache_path or self._cache_path)

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        if (cache_path is not None) and cache_path.is_dir():
            for log_path in cache_path.iterdir():
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _get_reward_path(self, token_path: Path) -> Path:
        """Get the reward cache path for a given token_path, respecting reward_cache_path override."""
        if self._reward_cache_path is not None:
            rel = token_path.relative_to(self._cache_path)
            return self._reward_cache_path / rel / "transfuser_reward.gz"
        return token_path / "transfuser_reward.gz"

    def _cache_scene_with_token(self, token: str) -> None:
        """
        Helper function to compute feature / targets and save in cache.
        :param token: unique identifier of scene to cache
        """

        scene = self._scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()

        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)

        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_targets(scene)
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper function to load feature / targets from cache.
        :param token:  unique identifier of scene to load
        :return: tuple of feature and target dictionaries
        """

        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        # features["token"] = token
        # token_path = self._valid_cache_paths[token]
        # features["token_path"] = str(token_path)
        features["token"] = token

        if self.train:
            reward_path = self._get_reward_path(token_path)
            if reward_path.exists():
                features["reward"] = load_feature_target_from_pickle(reward_path)
        return (features, targets)

    def cache_dataset(self) -> None:
        """Caches complete dataset into cache folder."""

        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        os.makedirs(self._cache_path, exist_ok=True)

        # determine tokens to cache
        if self._force_cache_computation:
            tokens_to_cache = self._scene_loader.tokens
        else:
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        :return: number of samples to load
        """
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get features or targets either from cache or computed on-the-fly.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """

        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        if self._cache_path is not None:
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            features, targets = self._load_scene_with_token(token)
        else:
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                targets.update(builder.compute_targets(scene))

        features["token"] = token
        return (features, targets)


class SequentialClipDataset(torch.utils.data.Dataset):
    """Wraps a CacheOnlyDataset and reorders tokens by temporal clip order.

    Loads a clip index (from analyze_scene_clips.py) and builds a flat token
    sequence: clip_0 tokens in order, clip_1 tokens in order, etc.
    Tokens not in any clip ("orphans") are appended at the end.

    Also exposes clip_boundaries for use with ClipParallelBatchSampler.
    """

    def __init__(
        self,
        base_dataset: CacheOnlyDataset,
        clip_index_path: str,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self._valid_tokens: Set[str] = set(base_dataset.tokens)

        # Load clip index
        with open(clip_index_path, "rb") as f:
            clip_data = pickle.load(f)
        raw_clips = clip_data["clips"]

        # Filter clips to valid tokens and drop empty clips
        self._clips: List[List[str]] = []
        used_tokens: Set[str] = set()
        for clip in raw_clips:
            filtered = [t for t in clip["tokens"] if t in self._valid_tokens]
            if filtered:
                self._clips.append(filtered)
                used_tokens.update(filtered)

        # Sort clips longest-first for better batch utilization
        self._clips.sort(key=len, reverse=True)

        # Orphan tokens not in any clip
        orphans = [t for t in base_dataset.tokens if t not in used_tokens]

        # Build flat token list and clip boundaries
        self.tokens: List[str] = []
        self.clip_boundaries: List[Tuple[int, int]] = []  # (start, end) per clip
        for clip_tokens in self._clips:
            start = len(self.tokens)
            self.tokens.extend(clip_tokens)
            self.clip_boundaries.append((start, len(self.tokens)))

        # Orphans as a pseudo-clip at the end
        if orphans:
            start = len(self.tokens)
            self.tokens.extend(orphans)
            self.clip_boundaries.append((start, len(self.tokens)))

        # Token -> base dataset index for O(1) lookup
        self._token_to_base_idx = {t: i for i, t in enumerate(base_dataset.tokens)}

        # Flat index -> (clip_idx, frame_in_clip) for O(1) lookup
        self._idx_to_clip_info: List[Tuple[int, int]] = [(-1, -1)] * len(self.tokens)
        for ci, (start, end) in enumerate(self.clip_boundaries):
            for flat_idx in range(start, end):
                self._idx_to_clip_info[flat_idx] = (ci, flat_idx - start)

        logger.info(
            "SequentialClipDataset: %d clips, %d orphans, %d total tokens",
            len(self._clips), len(orphans), len(self.tokens),
        )

    @property
    def num_clips(self) -> int:
        return len(self.clip_boundaries)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # Padding slot: load a real sample but mark token as None so the
        # inference loop knows to skip it.
        is_padding = (idx < 0)
        if is_padding:
            idx = 0  # load any valid sample for tensor shape

        token = self.tokens[idx]
        base_idx = self._token_to_base_idx[token]
        features, targets = self.base_dataset[base_idx]

        if is_padding:
            features["token"] = None
            features["clip_idx"] = -1
            features["frame_in_clip"] = -1
        else:
            clip_idx, frame_in_clip = self._idx_to_clip_info[idx]
            features["clip_idx"] = clip_idx
            features["frame_in_clip"] = frame_in_clip
        return features, targets


class ClipParallelBatchSampler(Sampler):
    """BatchSampler that steps through clips in lockstep.

    At each step, yields indices for frame `t` across all active clips.
    A clip becomes inactive once all its frames are exhausted.
    This ensures frame t+1 of any clip is only in a batch *after* frame t
    has been processed.

    Every batch is padded to exactly ``max_batch_size`` by repeating existing
    samples in the batch so that accelerate's even-batch logic works correctly
    with multi-GPU.  The inference loop should de-duplicate by token after
    gathering.

    Args:
        clip_boundaries: list of (start_idx, end_idx) from SequentialClipDataset
        max_batch_size: cap on how many clips to process in parallel.
            If None, all clips are active simultaneously.
    """

    def __init__(
        self,
        clip_boundaries: List[Tuple[int, int]],
        max_batch_size: Optional[int] = None,
    ):
        self.clip_boundaries = clip_boundaries
        self.max_batch_size = max_batch_size or len(clip_boundaries)
        self.batch_size = self.max_batch_size

    def __iter__(self):
        clips = list(self.clip_boundaries)
        bs = self.max_batch_size

        for chunk_start in range(0, len(clips), bs):
            active = clips[chunk_start : chunk_start + bs]
            max_len = max(end - start for start, end in active)

            for step in range(max_len):
                batch = []
                for start, end in active:
                    global_idx = start + step
                    if global_idx < end:
                        batch.append(global_idx)
                    else:
                        batch.append(-1)  # padding sentinel

                if not batch:
                    continue

                # Pad to exactly bs (for chunks smaller than bs)
                while len(batch) < bs:
                    batch.append(-1)

                yield batch

    def __len__(self):
        clips = list(self.clip_boundaries)
        bs = self.max_batch_size
        total = 0
        for chunk_start in range(0, len(clips), bs):
            active = clips[chunk_start : chunk_start + bs]
            total += max(end - start for start, end in active)
        return total
