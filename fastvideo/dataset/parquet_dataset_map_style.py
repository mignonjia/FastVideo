# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import pickle
import random
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
# Torch in general
import torch
import tqdm
# Dataset
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from fastvideo.platforms import current_platform

from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.distributed import (get_sp_world_size, get_world_group,
                                   get_world_rank, get_world_size)
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class DP_SP_BatchSampler(Sampler[list[int]]):
    """
    A simple sequential batch sampler that yields batches of indices.
    """

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        num_sp_groups: int,
        sp_world_size: int,
        global_rank: int,
        drop_last: bool = True,
        drop_first_row: bool = False,
        reshuffle_each_epoch: bool = True,
        seed: int = 0,
        candidate_indices: list[int] | None = None,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size
        self.drop_first_row = drop_first_row
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.candidate_indices = (torch.as_tensor(candidate_indices,
                                                  dtype=torch.long)
                                  if candidate_indices is not None else None)

        self._build_indices(0)  # build indices for epoch 0 to initialize the sampler

    def _build_indices(self, epoch: int) -> None:
        # ── epoch-level RNG ────────────────────────────────────────────────
        rng = torch.Generator().manual_seed(self.seed + epoch)
        if self.candidate_indices is None:
            global_indices = torch.randperm(self.dataset_size, generator=rng)
        else:
            perm = torch.randperm(len(self.candidate_indices), generator=rng)
            global_indices = self.candidate_indices[perm]

        dataset_size = len(global_indices)
        if self.drop_first_row:
            # drop 0 in global_indices
            global_indices = global_indices[global_indices != 0]
            dataset_size = len(global_indices)

        if self.drop_last:
            # For drop_last=True, we:
            # 1. Ensure total samples is divisible by (batch_size * num_sp_groups)
            # 2. This guarantees each SP group gets same number of complete batches
            # 3. Prevents uneven batch sizes across SP groups at end of epoch
            num_batches = dataset_size // self.batch_size
            num_global_batches = num_batches // self.num_sp_groups
            global_indices = global_indices[:num_global_batches *
                                            self.num_sp_groups *
                                            self.batch_size]
        else:
            if dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                # add more indices to make it divisible by (batch_size * num_sp_groups)
                padding_size = self.num_sp_groups * self.batch_size - (
                    dataset_size % (self.num_sp_groups * self.batch_size))
                logger.info("Padding the dataset from %d to %d",
                            dataset_size, dataset_size + padding_size)
                global_indices = torch.cat(
                    [global_indices, global_indices[:padding_size]])

        # shard the indices to each sp group
        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.
                                                num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d",
                    len(sp_group_local_indices))

    def set_candidate_indices(self,
                              candidate_indices: list[int] | None,
                              epoch: int = 0) -> None:
        self.candidate_indices = (torch.as_tensor(candidate_indices,
                                                  dtype=torch.long)
                                  if candidate_indices is not None else None)
        self._build_indices(epoch)

    def set_epoch(self, epoch: int) -> None:
        if not self.reshuffle_each_epoch:
            return
        self._build_indices(epoch)

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


def _parse_data_path_specs(path: str) -> list[tuple[str, int]]:
    """
    Parse data_path into a list of (directory, repeat_count).
    Syntax: comma-separated entries; each entry is "path" (default 1) or "path:N" (N = repeat count).
    N=0 means skip that path (convenience to disable without removing). If no ":" present, default is 1.
    Example: "/dir1:2,/dir2,/dir3:0" -> dir1 2x, dir2 1x, dir3 skipped.
    """
    specs: list[tuple[str, int]] = []
    for part in path.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            p, _, count_str = part.rpartition(":")
            p = p.strip()
            try:
                count = int(count_str.strip())
            except ValueError:
                raise ValueError(
                    f"data_path repeat count must be an integer, got {count_str!r}"
                ) from None
            if count < 0:
                raise ValueError(
                    f"data_path repeat count must be >= 0, got {count}"
                )
            specs.append((p, count))
        else:
            specs.append((part, 1))
    return specs


def _scan_parquet_files_for_path(p: str) -> tuple[list[str], list[int]]:
    """Return (file_paths, row_lengths) for a single directory."""
    file_names: list[str] = []
    for root, _, files in os.walk(p):
        for file in sorted(files):
            if file.endswith(".parquet"):
                file_names.append(os.path.join(root, file))
    lengths = []
    for file_path in tqdm.tqdm(
            file_names, desc="Reading parquet files to get lengths"):
        lengths.append(pq.ParquetFile(file_path).metadata.num_rows)
    logger.info("Found %d parquet files with %d total rows", len(file_names), sum(lengths))
    return file_names, lengths


def get_parquet_files_and_length(path: str):
    # ===== Custom block (wangame): support multi data_path entries with repeat counts =====
    # Accepted formats:
    #   "/path/a"
    #   "/path/a,/path/b"
    #   "/path/a:2,/path/b:1,/path/c:0"
    path_specs = _parse_data_path_specs(path)
    if not path_specs:
        raise ValueError(
            "data_path must be a non-empty path or comma-separated path specs")

    resolved_path_specs = [
        (os.path.realpath(os.path.expanduser(p)), count)
        for p, count in path_specs
    ]
    active_roots = [p for p, c in resolved_path_specs if c > 0]
    if not active_roots:
        raise ValueError("No active dataset paths after parsing data_path")

    is_single_no_repeat = (len(resolved_path_specs) == 1
                           and resolved_path_specs[0][1] == 1)
    if is_single_no_repeat:
        dataset_root = active_roots[0]
        cache_dir = os.path.join(dataset_root, "map_style_cache")
        cache_file = os.path.join(cache_dir, "file_info.pkl")
    else:
        # Multi-path/repeat mode uses a neutral cache keyed by full path spec.
        dataset_root = active_roots[0]
        neutral_root = os.environ.get(
            "FASTVIDEO_MAP_STYLE_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "fastvideo",
                         "map_style_cache"),
        )
        cache_dir = neutral_root
        cache_file = os.path.join(
            cache_dir,
            "file_info_" + hashlib.md5(path.strip().encode()).hexdigest()[:16]
            + ".pkl",
        )

    def _is_under_active_roots(file_path: str) -> bool:
        return any(
            os.path.commonpath([root, file_path]) == root
            for root in active_roots)
    # ===== End custom block (wangame) =====

    # Only rank 0 checks for cache and scans files if needed
    if get_world_rank() == 0:
        cache_loaded = False
        file_names_sorted = None
        lengths_sorted = None

        # First try to load existing cache
        if os.path.exists(cache_file):
            logger.info("Loading cached file info from %s", cache_file)
            try:
                with open(cache_file, "rb") as f:
                    file_names_sorted, lengths_sorted = pickle.load(f)
                file_names_sorted = tuple(
                    os.path.realpath(
                        os.path.join(os.getcwd(), p)
                        if not os.path.isabs(p) else p)
                    for p in file_names_sorted)
                files_outside_dataset_root = [
                    file_path for file_path in file_names_sorted
                    if not _is_under_active_roots(file_path)
                ]
                missing_files = [
                    file_path for file_path in file_names_sorted
                    if not os.path.exists(file_path)
                ]
                if files_outside_dataset_root:
                    logger.warning(
                        "Cached parquet file list points outside active dataset roots "
                        "(%s). Cache will be rebuilt. First out-of-root file: %s",
                        active_roots,
                        files_outside_dataset_root[0],
                    )
                    cache_loaded = False
                elif missing_files:
                    logger.warning(
                        "Cached parquet file list contains %d missing files. "
                        "Cache will be rebuilt. First missing file: %s",
                        len(missing_files),
                        missing_files[0],
                    )
                    cache_loaded = False
                else:
                    cache_loaded = True
                    logger.info("Successfully loaded cached file info")
            except Exception as e:
                logger.error("Error loading cached file info: %s", str(e))
                logger.info("Falling back to scanning files")
                cache_loaded = False

        # If cache not loaded (either doesn't exist or failed to load), scan files
        if not cache_loaded:
            if is_single_no_repeat:
                logger.info("Scanning parquet files to get lengths")
                lengths = []
                file_names = []
                for root, _, files in os.walk(dataset_root):
                    for file in sorted(files):
                        if file.endswith('.parquet'):
                            file_path = os.path.realpath(os.path.join(root, file))
                            file_names.append(file_path)
                if len(file_names) == 0:
                    raise FileNotFoundError(
                        "No parquet files found under dataset path: "
                        f"{path}. "
                        "Please verify this path points to preprocessed parquet "
                        "data.")
                for file_path in tqdm.tqdm(
                        file_names, desc="Reading parquet files to get lengths"):
                    num_rows = pq.ParquetFile(file_path).metadata.num_rows
                    lengths.append(num_rows)
                # sort according to file name to ensure all rank has the same order
                file_names_sorted, lengths_sorted = zip(*sorted(
                    zip(file_names, lengths, strict=True),
                    key=lambda x: x[0]),
                                                        strict=True)
            else:
                # ===== Custom block (wangame): multi-path scan and weighted repeats =====
                logger.info("Scanning parquet files (path specs: %s)",
                            [(p, c) for p, c in resolved_path_specs])
                combined: list[tuple[str, int, int]] = []
                sort_index = 0
                for p, count in resolved_path_specs:
                    if count == 0:
                        continue
                    fnames, lens = _scan_parquet_files_for_path(p)
                    for _ in range(count):
                        for f, ln in zip(fnames, lens, strict=True):
                            combined.append((os.path.realpath(f), ln, sort_index))
                            sort_index += 1
                if not combined:
                    raise FileNotFoundError(
                        "No parquet files found under active dataset paths: "
                        f"{active_roots}")
                combined.sort(key=lambda x: (x[0], x[2]))
                file_names_sorted = tuple(x[0] for x in combined)
                lengths_sorted = tuple(x[1] for x in combined)
                # ===== End custom block (wangame) =====

            # Save the cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump((file_names_sorted, lengths_sorted), f)
            logger.info("Saved file info to %s", cache_file)

    # Wait for rank 0 to finish creating/loading cache
    world_group = get_world_group()
    world_group.barrier()

    # Now all ranks load the cache (it should exist and be valid now)
    logger.info("Loading cached file info from %s after barrier", cache_file)
    with open(cache_file, "rb") as f:
        file_names_sorted, lengths_sorted = pickle.load(f)
    file_names_sorted = tuple(
        os.path.realpath(
            os.path.join(os.getcwd(), p) if not os.path.isabs(p) else p)
        for p in file_names_sorted)
    if len(file_names_sorted) == 0:
        raise RuntimeError(
            "Cached parquet metadata is empty after synchronization at "
            f"{cache_file}. "
            "Please verify the dataset path and regenerate cache.")
    if len(file_names_sorted) != len(lengths_sorted):
        raise RuntimeError(
            "Cached parquet metadata is corrupted at "
            f"{cache_file}: file count and length count do not match.")

    return file_names_sorted, lengths_sorted


def read_row_from_parquet_file(parquet_files: list[str], global_row_idx: int,
                               lengths: list[int]) -> dict[str, Any]:
    '''
    Read a row from a parquet file.
    Args:
        parquet_files: List[str]
        global_row_idx: int
        lengths: List[int]
    Returns:
    '''
    # find the parquet file and local row index
    cumulative = 0
    file_index = 0
    local_row_idx = 0

    for file_index in range(len(lengths)):
        if cumulative + lengths[file_index] > global_row_idx:
            local_row_idx = global_row_idx - cumulative
            break
        cumulative += lengths[file_index]
    else:
        # If we reach here, global_row_idx is out of bounds
        raise IndexError(
            f"global_row_idx {global_row_idx} is out of bounds for dataset")

    parquet_file = pq.ParquetFile(parquet_files[file_index])

    # Calculate the row group to read into memory and the local idx
    # This way we can avoid reading in the entire parquet file
    cumulative = 0
    row_group_index = 0
    local_index = 0

    for i in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(i).num_rows
        if cumulative + num_rows > local_row_idx:
            row_group_index = i
            local_index = local_row_idx - cumulative
            break
        cumulative += num_rows
    else:
        # If we reach here, local_row_idx is out of bounds for this parquet file
        raise IndexError(
            f"local_row_idx {local_row_idx} is out of bounds for parquet file {parquet_files[file_index]}"
        )

    row_group = parquet_file.read_row_group(row_group_index).to_pydict()
    row_dict = {k: v[local_index] for k, v in row_group.items()}
    del row_group

    return row_dict


# ────────────────────────────────────────────────────────────────────────────
# 2.  Dataset with batched __getitems__
# ────────────────────────────────────────────────────────────────────────────
class LatentsParquetMapStyleDataset(Dataset):
    """
    Return latents[B,C,T,H,W] and embeddings[B,L,D] in pinned CPU memory.
    Note: 
    Using parquet for map style dataset is not efficient, we mainly keep it for backward compatibility and debugging.
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        parquet_schema: pa.Schema,
        cfg_rate: float = 0.0,
        seed: int = 42,
        drop_last: bool = True,
        drop_first_row: bool = False,
        reshuffle_each_epoch: bool = False,
        text_padding_length: int = 512,
        candidate_indices: list[int] | None = None,
    ):
        super().__init__()
        self.path = path
        self.cfg_rate = cfg_rate
        self.parquet_schema = parquet_schema
        self.seed = seed
        # Create a seeded random generator for deterministic CFG
        self.rng = random.Random(seed)
        logger.info("Initializing LatentsParquetMapStyleDataset with path: %s",
                    path)
        self.parquet_files, self.lengths = get_parquet_files_and_length(path)
        self.batch = batch_size
        self.text_padding_length = text_padding_length
        self.sampler = DP_SP_BatchSampler(
            batch_size=batch_size,
            dataset_size=sum(self.lengths),
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            reshuffle_each_epoch=reshuffle_each_epoch,
            seed=seed,
            candidate_indices=candidate_indices,
        )
        logger.info("Dataset initialized with %d parquet files and %d rows",
                    len(self.parquet_files), sum(self.lengths))

    def get_validation_negative_prompt(
            self) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get the negative prompt for validation. 
        This method ensures the negative prompt is loaded and cached properly.
        Returns the processed negative prompt data (latents, embeddings, masks, info).
        """

        # Read first row from first parquet file
        file_path = self.parquet_files[0]
        row_idx = 0
        # Read the negative prompt data
        row_dict = read_row_from_parquet_file([file_path], row_idx,
                                              [self.lengths[0]])

        batch = collate_rows_from_parquet_schema([row_dict],
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=0.0,
                                                 rng=self.rng)
        negative_prompt = batch['info_list'][0]['prompt']
        negative_prompt_embedding = batch['text_embedding']
        negative_prompt_attention_mask = batch['text_attention_mask']
        if len(negative_prompt_embedding.shape) == 2:
            negative_prompt_embedding = negative_prompt_embedding.unsqueeze(0)
        if len(negative_prompt_attention_mask.shape) == 1:
            negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(
                0).unsqueeze(0)

        return negative_prompt_embedding, negative_prompt_attention_mask, negative_prompt

    # PyTorch calls this ONLY because the batch_sampler yields a list
    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        """
        Batch fetch using read_row_from_parquet_file for each index.
        """
        rows = [
            read_row_from_parquet_file(self.parquet_files, idx, self.lengths)
            for idx in indices
        ]

        batch = collate_rows_from_parquet_schema(rows,
                                                 self.parquet_schema,
                                                 self.text_padding_length,
                                                 cfg_rate=self.cfg_rate,
                                                 rng=self.rng)
        return batch

    def __len__(self):
        return sum(self.lengths)


# ────────────────────────────────────────────────────────────────────────────
# 3.  Loader helper – everything else stays just like your original trainer
# ────────────────────────────────────────────────────────────────────────────
def passthrough(batch):
    return batch


def build_bot_died_excluded_indices(
    data_path: str,
    parquet_files: list[str],
    lengths: list[int],
) -> set[int]:
    """Scan data_path dirs for ``../filter/bot_died.json`` relative to each
    preprocessed directory.  When found, read the ``file_name`` column from
    parquet files belonging to that directory and return global row indices
    whose ``int(file_name)`` appears in the bot_died list.
    """
    import json

    path_specs = _parse_data_path_specs(data_path)

    bot_died_per_dir: dict[str, set[int]] = {}
    for p, count in path_specs:
        if count == 0:
            continue
        filter_path = os.path.join(os.path.dirname(p), "filter", "bot_died.json")
        if os.path.exists(filter_path):
            with open(filter_path) as f:
                bot_died_per_dir[p] = set(json.load(f))
            logger.info(
                "Loaded bot_died filter from %s: %d entries to exclude",
                filter_path, len(bot_died_per_dir[p]))

    if not bot_died_per_dir:
        return set()

    excluded: set[int] = set()
    global_offset = 0
    for file_path, length in zip(parquet_files, lengths, strict=True):
        matching_dir = None
        for dir_path in bot_died_per_dir:
            if file_path.startswith(dir_path):
                matching_dir = dir_path
                break

        if matching_dir is not None:
            bot_died_set = bot_died_per_dir[matching_dir]
            try:
                table = pq.read_table(file_path, columns=["file_name"])
                file_names = table.column("file_name").to_pylist()
                for local_idx, fn in enumerate(file_names):
                    if int(str(fn).strip()) in bot_died_set:
                        excluded.add(global_offset + local_idx)
            except Exception as e:
                logger.warning(
                    "Failed to read file_name from %s for bot_died filter: %s",
                    file_path, e)

        global_offset += length

    return excluded


def build_parquet_map_style_dataloader(
        path,
        batch_size,
        num_data_workers,
        parquet_schema,
        cfg_rate=0.0,
        drop_last=True,
        drop_first_row=False,
        reshuffle_each_epoch=False,
        text_padding_length=512,
        seed=42,
        candidate_indices: list[int] | None = None
) -> tuple[LatentsParquetMapStyleDataset, StatefulDataLoader]:
    dataset = LatentsParquetMapStyleDataset(
        path,
        batch_size,
        cfg_rate=cfg_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        reshuffle_each_epoch=reshuffle_each_epoch,
        text_padding_length=text_padding_length,
        parquet_schema=parquet_schema,
        seed=seed,
        candidate_indices=candidate_indices)

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=passthrough,
        num_workers=num_data_workers,
        pin_memory=True,
        pin_memory_device=current_platform.device_name,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
