"""
ref: https://github.com/volcengine/verl/blob/main/verl/protocol.py
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import copy
import io
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set, TYPE_CHECKING

import numpy as np
import ray
import tensordict
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from roll.utils.functionals import union_two_dict, divide_by_chunk_size
from roll.platforms import current_platform
from roll.utils.logging import get_logger

logger = get_logger()

try:
    tensordict.set_lazy_legacy(False).set()
except:
    pass

TQ_AVAILABLE = False
try:
    import transfer_queue as tq
    from transfer_queue import BatchMeta, KVBatchMeta
    TQ_AVAILABLE = True
except ImportError:
    BatchMeta = None
    KVBatchMeta = None

if TYPE_CHECKING:
    try:
        from transfer_queue import BatchMeta, KVBatchMeta
    except ImportError:
        BatchMeta = Any
        KVBatchMeta = Any


def _is_tq_enabled() -> bool:
    return TQ_AVAILABLE and os.getenv("ROLL_ENABLE_TQ", "0") == "1"


def _get_tq_partition_id() -> str:
    return os.getenv("ROLL_TQ_PARTITION_ID", "default")


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert (
        tensor_dict1.batch_size == tensor_dict2.batch_size
    ), f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(
                tensor_dict2[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            assert np.all(
                tensor_dict2[key] == tensor_dict1[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    """
    Convert a list of dictionaries into a dictionary of lists.

    Example:
        Input:  [{"a": 1, "b": 2}, {"a": 3}, {"b": 4}]
        Output: {"a": [1, 3], "b": [2, 4]}

    Only keys present in each dictionary are aggregated.
    Missing keys in a dictionary are simply skipped.
    """
    if not list_of_dict:
        return {}

    output = {}
    for d in list_of_dict:
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, but got {type(d)}: {d}")
        for k, v in d.items():
            output.setdefault(k, []).append(v)

    return output


def collate_fn(x: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    meta_info = None
    for data in x:
        meta_info = data.meta_info
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.empty(len(val), dtype=object)
        non_tensor_batch[key][:] = val
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def move_tensors_to_device(data, device):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = move_tensors_to_device(val, device)
    elif isinstance(data, list):
        for index, val in enumerate(data):
            data[index] = move_tensors_to_device(val, device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def custom_np_concatenate(val):
    concatenated_list = []
    for array in val:
        concatenated_list.extend(array)
    concatenated_array = np.empty(len(concatenated_list), dtype=object)
    concatenated_array[:] = concatenated_list
    return concatenated_array


@dataclass
class DataProtoItem:
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()
    
        if self.batch is not None and current_platform.is_npu():
            for key, val in self.batch.items():
                if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    self.batch[key] = val.to(torch.int32)

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        if self.non_tensor_batch is not None:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0

    def __getitem__(self, item):
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            DataProto: For all indexing types except single integers
            DataProtoItem: Only for single integer indices
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        elif isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item]
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        if _is_tq_enabled() and self.batch is not None and len(self) > 0:
            return self._getstate_tq()
        return self._getstate_pickle()

    def _getstate_pickle(self):
        buffer = io.BytesIO()
        if tensordict.__version__ >= "0.5.0" and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        torch.save(self.batch, buffer)
        return ("pickle", buffer, self.non_tensor_batch, self.meta_info)

    def _getstate_tq(self):
        import time
        t1 = time.time()
        
        tq.init()
        tq_client = tq.get_client()
        partition_id = _get_tq_partition_id()
        batch_size = len(self)
        
        meta = tq_client.put(
            data=self.batch,
            metadata=BatchMeta(
                global_indexes=list(range(batch_size)),
                partition_ids=[partition_id] * batch_size,
            )
        )
        
        extra_info = {}
        if self.non_tensor_batch:
            extra_info["non_tensor_batch"] = copy.deepcopy(self.non_tensor_batch)
        if self.meta_info:
            extra_info["meta_info"] = copy.deepcopy(self.meta_info)
        meta.extra_info = extra_info
        
        t2 = time.time()
        logger.debug(f"DataProto serialized to TransferQueue, size={batch_size}, cost={t2-t1:.3f}s")
        
        return ("tq", meta)

    def __setstate__(self, data):
        if isinstance(data, tuple) and len(data) >= 1:
            if data[0] == "tq":
                self._setstate_tq(data[1])
                return
            elif data[0] == "pickle":
                self._setstate_pickle(data[1], data[2], data[3])
                return
        
        self._setstate_pickle(data[0], data[1], data[2])

    def _setstate_pickle(self, batch_buffer, non_tensor_batch, meta_info):
        batch_buffer.seek(0)
        batch = torch.load(
            batch_buffer, weights_only=False, map_location="cpu" if not current_platform.is_available() else None
        )
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch if non_tensor_batch is not None else {}
        self.meta_info = meta_info if meta_info is not None else {}

    def _setstate_tq(self, meta: "BatchMeta"):
        import time
        t1 = time.time()
        
        tq_client = tq.get_client()
        tensordict = tq_client.get_data(meta)
        
        extra_info = meta.extra_info or {}
        non_tensor_batch = extra_info.get("non_tensor_batch", {})
        meta_info = extra_info.get("meta_info", {})
        
        self.batch = tensordict
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info
        
        t2 = time.time()
        logger.debug(f"DataProto deserialized from TransferQueue, size={len(self)}, cost={t2-t1:.3f}s")

    def to_tq_meta(self, partition_id: Optional[str] = None) -> Optional["BatchMeta"]:
        """
        Convert DataProto to TransferQueue BatchMeta for efficient transfer.
        
        Args:
            partition_id: Optional partition ID for TransferQueue.
                         If not provided, uses ROLL_TQ_PARTITION_ID env var or "default".
        
        Returns:
            BatchMeta if TransferQueue is available and data is valid, None otherwise.
        """
        if not TQ_AVAILABLE:
            logger.warning("TransferQueue is not available, cannot convert to BatchMeta")
            return None
        
        if self.batch is None or len(self) == 0:
            return None
        
        tq.init()
        tq_client = tq.get_client()
        partition_id = partition_id or _get_tq_partition_id()
        batch_size = len(self)
        
        meta = tq_client.put(
            data=self.batch,
            metadata=BatchMeta(
                global_indexes=list(range(batch_size)),
                partition_ids=[partition_id] * batch_size,
            )
        )
        
        extra_info = {}
        if self.non_tensor_batch:
            extra_info["non_tensor_batch"] = copy.deepcopy(self.non_tensor_batch)
        if self.meta_info:
            extra_info["meta_info"] = copy.deepcopy(self.meta_info)
        meta.extra_info = extra_info
        
        return meta

    @classmethod
    def from_tq_meta(cls, meta: "BatchMeta") -> "DataProto":
        """
        Create DataProto from TransferQueue BatchMeta.
        
        Args:
            meta: BatchMeta from TransferQueue.
        
        Returns:
            DataProto with data retrieved from TransferQueue.
        """
        if not TQ_AVAILABLE:
            raise RuntimeError("TransferQueue is not available")
        
        tq_client = tq.get_client()
        tensordict = tq_client.get_data(meta)
        
        extra_info = meta.extra_info or {}
        non_tensor_batch = extra_info.get("non_tensor_batch", {})
        meta_info = extra_info.get("meta_info", {})
        
        return cls(batch=tensordict, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    @property
    def is_tq_backed(self) -> bool:
        """Check if this DataProto is backed by TransferQueue data."""
        return hasattr(self, '_tq_meta') and self._tq_meta is not None

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert (
                    isinstance(val, np.ndarray) and val.dtype == object
                ), "data in the non_tensor_batch must be a numpy.array with dtype=object"
                assert (
                    val.shape[0] == batch_size
                ), f"key {key} length {len(val)} is not equal to batch size {batch_size}"

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if current_platform.is_npu() and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    val = val.to(torch.int32)
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        assert len(tensors) > 0, "tensors must not be empty"
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert (
                    batch_size == current_batch
                ), f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"

        for key, val in non_tensors.items():
            non_tensors[key] = np.empty(len(val), dtype=object)
            non_tensors[key][:] = val

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        if self.meta_info is not None:
            self.meta_info = move_tensors_to_device(self.meta_info, device)

        return self

    def clone(self) -> "DataProto":
        """
        Create a deep copy of this DataProto, including tensors,
        non-tensor data, and meta_info.

        The new DataProto will share no underlying storage with the original.

        Returns:
            DataProto: A new DataProto instance with the same content but
                       independent memory.
        """
        # Copy batch
        batch_copy = self.batch.clone() if self.batch is not None else None

        # Copy non-tensor objects (numpy arrays)
        non_tensor_copy = {k: np.copy(v) for k, v in self.non_tensor_batch.items()}

        # Deep copy meta_info to avoid shared mutable objects
        meta_copy = copy.deepcopy(self.meta_info)

        # Return new DataProto instance
        return DataProto(
            batch=batch_copy,
            non_tensor_batch=non_tensor_copy,
            meta_info=meta_copy
        )

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()}, batch_size=(batch_size,)
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        Slice the DataProto and return a new DataProto object.
        This is an improved version of direct slicing which returns a DataProtoItem.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index still returns DataProtoItem
            single_item = data_proto[5]
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []
        batch_keys = self.validate_input(batch_keys)
        non_tensor_batch_keys = self.validate_input(non_tensor_batch_keys)
        meta_info_keys = self.validate_input(meta_info_keys)

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @staticmethod
    def validate_input(keys):
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif isinstance(keys, list):
                pass
            else:
                raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
        return keys

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        Note that this function only rename the key in the batch
        """

        old_keys = self.validate_input(old_keys)
        new_keys = self.validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if
        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.
        要求:
            batch_size > chunks，调用方保证，此处保证每个chunk会返回一个DataProto

        np.array_split(val, chunks) 和 self.batch.chunk(chunks=chunks, dim=0) 在不能均分时行为不同
        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        chunks_sizes = None
        if len(self) > 0:
            assert len(self) >= chunks, f"batch_size {self.batch.batch_size[0]} < chunks {chunks}"
            index_array = np.arange(len(self))
            chunks_sizes = [len(b) for b in np.array_split(index_array, chunks)]

        if self.batch is not None:
            batch_lst = divide_by_chunk_size(self.batch, chunk_sizes=chunks_sizes)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            non_tensor_lst = divide_by_chunk_size(val, chunk_sizes=chunks_sizes)
            assert len(non_tensor_lst) == chunks, f"len(non_tensor_lst) {len(non_tensor_lst)} != chunks {chunks}"
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                DataProto(
                    batch=batch_lst[i].clone() if batch_lst[i] is not None else batch_lst[i],
                    non_tensor_batch=non_tensor_batch_lst[i],
                    meta_info=self.meta_info,
                )
            )

        return output

    @staticmethod
    def concat(
            data: List["DataProto"],
            *,
            global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        """
        Concatenate a list of DataProto objects.

        Parameters
        ----------
        data : List[DataProto]
            List of DataProto instances to be concatenated.
        global_keys : Set[str], optional
            Keys in `meta_info` that should be **aggregated across ranks**.
            - If the value is a dict, each sub-key is concatenated across ranks.
            - Otherwise, values are collected into a list.
            Keys not listed retain only the value from rank 0.

        Returns
        -------
        DataProto
            A new DataProto with concatenated tensors, non-tensor data,
            and processed meta information.
        """
        global_keys = global_keys if global_keys is not None else {"metrics"}

        # ---------- 1. Concatenate tensor / non-tensor batches ----------
        batch_lst = [d.batch for d in data if d.batch is not None]
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst else None

        non_tensor_batch = list_of_dict_to_dict_of_list(
            [d.non_tensor_batch for d in data]
        )
        for k, v in non_tensor_batch.items():
            non_tensor_batch[k] = custom_np_concatenate(v)

        # ---------- 2. Aggregate meta information ----------
        merged_meta = dict(data[0].meta_info)  # start with rank-0 values

        for key in global_keys:
            if key not in merged_meta:
                continue

            values = [d.meta_info.get(key) for d in data]

            # Case 1: dict — aggregate each sub-key across ranks
            if isinstance(merged_meta[key], dict):
                sub_dict = list_of_dict_to_dict_of_list(values)
                for sub_key, sub_list in sub_dict.items():
                    try:
                        if np.isscalar(sub_list[0]):
                            sub_dict[sub_key] = np.array(sub_list).tolist()
                        else:
                            sub_dict[sub_key] = np.concatenate(sub_list, axis=0).tolist()
                    except Exception:
                        # fallback: keep as list
                        sub_dict[sub_key] = sub_list
                merged_meta[key] = sub_dict

            # Case 2: non-dict — collect into list
            else:
                merged_meta[key] = values

        return DataProto(
            batch=new_batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=merged_meta,
        )

    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        # Ensure that indices is at least a 1-D tensor.
        indices = indices.view(-1) if indices.dim() == 0 else indices
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def group_by(self, keys: Union[List[str], str]) -> Dict[str, "DataProto"]:
        """
        Group the data by specified keys. Supports grouping by both tensor and non-tensor fields.

        Args:
            keys: Field names to group by. Can be either in batch (tensors) or non_tensor_batch

        Returns:
            Dictionary mapping group keys to DataProto instances containing matching data

        Example:
            Given data with field "category" having values ["A", "B", "A"],
            returns {"A": DataProto(A_data), "B": DataProto(B_data)}
        """
        keys = self.validate_input(keys)
        assert len(keys) > 0, "Must provide at least one grouping key"

        # Collect grouping values across data types
        group_key_values = []
        for idx in range(len(self)):
            key_values = []
            for key in keys:
                # Check tensor data first
                if key in self.batch.keys():
                    key_values.append(str(self.batch[key][idx].numpy()))
                elif key in self.non_tensor_batch:
                    key_values.append(str(self.non_tensor_batch[key][idx]))
                else:
                    raise KeyError(f"Grouping key '{key}' not found in tensor or non-tensor data")

            # Create composite key for multi-field grouping
            group_key = "|".join(key_values) if len(key_values) > 1 else key_values[0]
            group_key_values.append(group_key)

        # Create index groups
        groups = defaultdict(list)
        for idx, group_key in enumerate(group_key_values):
            groups[group_key].append(idx)

        # Create grouped DataProtos
        grouped_data = {}
        for group_key, indices in groups.items():
            grouped_data[group_key] = collate_fn([self[idx] for idx in indices])

        return grouped_data

    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    @staticmethod
    def materialize_concat(
            data_refs: Union[List[ray.ObjectRef], ray.ObjectRef, List["ObjectRefWrap"]],
            *,
            global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        """
        Fetch a collection of DataProto objects from Ray ObjectRef(s) and concatenate
        them into a single DataProto instance.

        Parameters
        ----------
        data_refs : Union[List[ray.ObjectRef], ray.ObjectRef, List[ObjectRefWrap]]
            Ray object references (or ObjectRefWrap) pointing to DataProto objects.
        global_keys : Optional[Set[str]], optional
            Keys in ``meta_info`` that should be aggregated across all ranks when
            concatenating.  If None, only rank-0 values are kept for all keys.

        Returns
        -------
        DataProto
            The concatenated DataProto instance.
        """
        # Normalize input to List[<reference>]
        if isinstance(data_refs, DataProto):
            data_refs = [data_refs]

        timeout = None
        if "roll_RPC_TIMEOUT" in os.environ:
            timeout = int(os.environ["roll_RPC_TIMEOUT"])

        # Fetch objects from Ray
        if isinstance(data_refs[0], ObjectRefWrap):
            data_refs: List[ObjectRefWrap]
            obj_refs = [ref.obj_ref for ref in data_refs]
            fetched = ray.get(obj_refs, timeout=timeout)
            data = [fetched[i] for i, ref in enumerate(data_refs) if ref.collected]
        else:
            data: List["DataProto"] = ray.get(data_refs, timeout=timeout)

        # Concatenate and apply global aggregation rules
        return DataProto.concat(data, global_keys=global_keys)


class ObjectRefWrap:
    def __init__(self, obj_ref: ray.ObjectRef, collected=False):
        self.obj_ref = obj_ref
        self.collected = collected


class LazyDataProto(DataProto):
    """
    A lazy-loading DataProto that integrates with TransferQueue for efficient data transfer.
    
    Key features:
    1. Transparent TransferQueue integration - users don't need to change any code
    2. Lazy loading - data is only loaded from TransferQueue when accessed
    3. Automatic fallback to pickle serialization when TransferQueue is not available
    4. 100% API compatible with DataProto
    
    Usage:
        # Same as DataProto - no changes needed
        data = LazyDataProto(batch=tensor_dict, non_tensor_batch=non_tensor, meta_info=meta)
        
        # When ROLL_ENABLE_TQ=1, serialization automatically uses TransferQueue
        # Otherwise, falls back to pickle
    """
    
    _tq_meta: "BatchMeta" = field(default=None, init=False)
    _materialized: bool = field(default=True, init=False)
    _batch: TensorDict = field(default=None, init=False)
    
    def __init__(
        self,
        batch: TensorDict = None,
        non_tensor_batch: Dict = None,
        meta_info: Dict = None,
        _tq_meta: "BatchMeta" = None,
    ):
        self._tq_meta = _tq_meta
        self._materialized = _tq_meta is None
        self._batch = batch
        
        if non_tensor_batch is None:
            non_tensor_batch = {}
        if meta_info is None:
            meta_info = {}
        
        object.__setattr__(self, 'non_tensor_batch', non_tensor_batch)
        object.__setattr__(self, 'meta_info', meta_info)
    
    def __post_init__(self):
        if self._materialized and self._batch is not None:
            if current_platform.is_npu():
                for key, val in self._batch.items():
                    if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                        logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                        self._batch[key] = val.to(torch.int32)
    
    def _materialize(self):
        """Load data from TransferQueue if not already materialized."""
        if self._materialized or self._tq_meta is None:
            return
        
        if not TQ_AVAILABLE:
            raise RuntimeError("TransferQueue is not available but _tq_meta is set")
        
        import time
        t1 = time.time()
        
        tq_client = tq.get_client()
        tensordict = tq_client.get_data(self._tq_meta)
        
        extra_info = self._tq_meta.extra_info or {}
        
        self._batch = tensordict
        if "non_tensor_batch" in extra_info:
            object.__setattr__(self, 'non_tensor_batch', extra_info["non_tensor_batch"])
        if "meta_info" in extra_info:
            object.__setattr__(self, 'meta_info', extra_info["meta_info"])
        
        self._materialized = True
        self._tq_meta = None
        
        t2 = time.time()
        logger.debug(f"LazyDataProto materialized, size={len(self)}, cost={t2-t1:.3f}s")
    
    def __len__(self):
        if self._tq_meta is not None and not self._materialized:
            return self._tq_meta.size
        if self._batch is not None:
            return self._batch.batch_size[0]
        if self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0
    
    def __getitem__(self, item):
        self._materialize()
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        if isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)
        if isinstance(item, int):
            return self._get_single_item(item)
        raise TypeError(f"Invalid index type: {type(item)}")
    
    def _get_single_item(self, idx: int):
        self._materialize()
        batch_item = self._batch[idx] if self._batch is not None else None
        non_tensor_item = {k: v[idx] for k, v in self.non_tensor_batch.items()}
        return DataProtoItem(batch=batch_item, non_tensor_batch=non_tensor_item, meta_info=self.meta_info)
    
    @property
    def batch(self):
        self._materialize()
        return self._batch
    
    @batch.setter
    def batch(self, value):
        self._batch = value
        self._materialized = True
    
    @property
    def is_materialized(self) -> bool:
        """Check if data has been loaded from TransferQueue."""
        return self._materialized
    
    @property
    def is_lazy(self) -> bool:
        """Check if this LazyDataProto is still in lazy mode (not materialized)."""
        return self._tq_meta is not None and not self._materialized
    
    def __getstate__(self):
        if self._tq_meta is not None and not self._materialized:
            return ("lazy_tq", self._tq_meta, self.non_tensor_batch, self.meta_info)
        
        if _is_tq_enabled() and self._batch is not None and len(self) > 0:
            return self._getstate_tq()
        return self._getstate_pickle()
    
    def _getstate_pickle(self):
        buffer = io.BytesIO()
        if tensordict.__version__ >= "0.5.0" and self._batch is not None:
            self._batch = self._batch.contiguous()
            self._batch = self._batch.consolidate()
        torch.save(self._batch, buffer)
        return ("pickle", buffer, self.non_tensor_batch, self.meta_info)
    
    def _getstate_tq(self):
        import time
        t1 = time.time()
        
        tq.init()
        tq_client = tq.get_client()
        partition_id = _get_tq_partition_id()
        batch_size = len(self)
        
        meta = tq_client.put(
            data=self._batch,
            metadata=BatchMeta(
                global_indexes=list(range(batch_size)),
                partition_ids=[partition_id] * batch_size,
            )
        )
        
        extra_info = {}
        if self.non_tensor_batch:
            extra_info["non_tensor_batch"] = copy.deepcopy(self.non_tensor_batch)
        if self.meta_info:
            extra_info["meta_info"] = copy.deepcopy(self.meta_info)
        meta.extra_info = extra_info
        
        t2 = time.time()
        logger.debug(f"LazyDataProto serialized to TransferQueue, size={batch_size}, cost={t2-t1:.3f}s")
        
        return ("tq", meta)
    
    def __setstate__(self, data):
        if isinstance(data, tuple) and len(data) >= 1:
            if data[0] == "lazy_tq":
                self._tq_meta = data[1]
                self._materialized = False
                self._batch = None
                self.non_tensor_batch = data[2] if len(data) > 2 else {}
                self.meta_info = data[3] if len(data) > 3 else {}
                return
            elif data[0] == "tq":
                self._tq_meta = data[1]
                self._materialized = False
                self._batch = None
                self.non_tensor_batch = {}
                self.meta_info = {}
                return
            elif data[0] == "pickle":
                self._setstate_pickle(data[1], data[2], data[3])
                return
        
        self._setstate_pickle(data[0], data[1], data[2])
    
    def _setstate_pickle(self, batch_buffer, non_tensor_batch, meta_info):
        batch_buffer.seek(0)
        batch = torch.load(
            batch_buffer, weights_only=False, map_location="cpu" if not current_platform.is_available() else None
        )
        self._batch = batch
        self.non_tensor_batch = non_tensor_batch if non_tensor_batch is not None else {}
        self.meta_info = meta_info if meta_info is not None else {}
        self._tq_meta = None
        self._materialized = True
    
    def materialize(self) -> "LazyDataProto":
        """Force materialization and return self for chaining."""
        self._materialize()
        return self
    
    def to_dataproto(self) -> DataProto:
        """Convert to a regular DataProto (forces materialization)."""
        self._materialize()
        return DataProto(
            batch=self._batch,
            non_tensor_batch=copy.deepcopy(self.non_tensor_batch),
            meta_info=copy.deepcopy(self.meta_info),
        )
    
    @classmethod
    def from_dataproto(cls, data: DataProto) -> "LazyDataProto":
        """Create a LazyDataProto from a DataProto."""
        return cls(
            batch=data.batch,
            non_tensor_batch=copy.deepcopy(data.non_tensor_batch),
            meta_info=copy.deepcopy(data.meta_info),
        )
    
    @classmethod
    def from_tq_meta(cls, meta: "BatchMeta") -> "LazyDataProto":
        """Create a lazy LazyDataProto from TransferQueue BatchMeta (no data loading)."""
        extra_info = meta.extra_info or {}
        return cls(
            batch=None,
            non_tensor_batch=extra_info.get("non_tensor_batch", {}),
            meta_info=extra_info.get("meta_info", {}),
            _tq_meta=meta,
        )
    
    def to_tq_meta(self, partition_id: Optional[str] = None) -> Optional["BatchMeta"]:
        """Convert to TransferQueue BatchMeta for efficient transfer."""
        if self._tq_meta is not None and not self._materialized:
            return self._tq_meta
        
        if not TQ_AVAILABLE:
            logger.warning("TransferQueue is not available")
            return None
        
        if self._batch is None or len(self) == 0:
            return None
        
        tq.init()
        tq_client = tq.get_client()
        partition_id = partition_id or _get_tq_partition_id()
        batch_size = len(self)
        
        meta = tq_client.put(
            data=self._batch,
            metadata=BatchMeta(
                global_indexes=list(range(batch_size)),
                partition_ids=[partition_id] * batch_size,
            )
        )
        
        extra_info = {}
        if self.non_tensor_batch:
            extra_info["non_tensor_batch"] = copy.deepcopy(self.non_tensor_batch)
        if self.meta_info:
            extra_info["meta_info"] = copy.deepcopy(self.meta_info)
        meta.extra_info = extra_info
        
        return meta
    
    def to(self, device) -> "LazyDataProto":
        self._materialize()
        if self._batch is not None:
            self._batch = self._batch.to(device)
        if self.meta_info is not None:
            self.meta_info = move_tensors_to_device(self.meta_info, device)
        return self
    
    def clone(self) -> "LazyDataProto":
        self._materialize()
        batch_copy = self._batch.clone() if self._batch is not None else None
        non_tensor_copy = {k: np.copy(v) for k, v in self.non_tensor_batch.items()}
        meta_copy = copy.deepcopy(self.meta_info)
        return LazyDataProto(
            batch=batch_copy,
            non_tensor_batch=non_tensor_copy,
            meta_info=meta_copy,
        )
    
    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "LazyDataProto":
        self._materialize()
        
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self._batch.select(*batch_keys)
        else:
            sub_batch = self._batch
        
        if non_tensor_batch_keys is not None:
            non_tensor_batch = {k: v for k, v in self.non_tensor_batch.items() if k in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch
        
        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)
        
        if meta_info_keys is not None:
            sub_meta_info = {k: v for k, v in self.meta_info.items() if k in meta_info_keys}
        else:
            sub_meta_info = self.meta_info
        
        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)
        
        return LazyDataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)
    
    def select_idxs(self, idxs) -> "LazyDataProto":
        self._materialize()
        
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)
        
        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()
        
        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]
        
        if self._batch is not None:
            selected_batch = TensorDict(
                source={k: t[idxs_torch] for k, t in self._batch.items()},
                batch_size=(batch_size,),
            )
        else:
            selected_batch = None
        
        selected_non_tensor = {k: v[idxs_np] for k, v in self.non_tensor_batch.items()}
        
        return LazyDataProto(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)
    
    def slice(self, start=None, end=None, step=None) -> "LazyDataProto":
        self._materialize()
        slice_obj = slice(start, end, step)
        
        if self._batch is not None:
            sliced_batch = self._batch[slice_obj]
        else:
            sliced_batch = None
        
        sliced_non_tensor = {k: v[slice_obj] for k, v in self.non_tensor_batch.items()}
        
        return LazyDataProto(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)
    
    def chunk(self, chunks: int) -> List["LazyDataProto"]:
        self._materialize()
        
        chunks_sizes = None
        if len(self) > 0:
            assert len(self) >= chunks, f"batch_size {len(self)} < chunks {chunks}"
            index_array = np.arange(len(self))
            chunks_sizes = [len(b) for b in np.array_split(index_array, chunks)]
        
        if self._batch is not None:
            batch_lst = divide_by_chunk_size(self._batch, chunk_sizes=chunks_sizes)
        else:
            batch_lst = [None for _ in range(chunks)]
        
        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            non_tensor_lst = divide_by_chunk_size(val, chunk_sizes=chunks_sizes)
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]
        
        output = []
        for i in range(chunks):
            output.append(
                LazyDataProto(
                    batch=batch_lst[i].clone() if batch_lst[i] is not None else batch_lst[i],
                    non_tensor_batch=non_tensor_batch_lst[i],
                    meta_info=self.meta_info,
                )
            )
        
        return output
    
    @staticmethod
    def concat(
        data: List["LazyDataProto"],
        *,
        global_keys: Optional[Set[str]] = None,
    ) -> "LazyDataProto":
        global_keys = global_keys if global_keys is not None else {"metrics"}
        
        for d in data:
            d._materialize()
        
        batch_lst = [d._batch for d in data if d._batch is not None]
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst else None
        
        non_tensor_batch = list_of_dict_to_dict_of_list([d.non_tensor_batch for d in data])
        for k, v in non_tensor_batch.items():
            non_tensor_batch[k] = custom_np_concatenate(v)
        
        merged_meta = dict(data[0].meta_info)
        
        for key in global_keys:
            if key not in merged_meta:
                continue
            values = [d.meta_info.get(key) for d in data]
            
            if isinstance(merged_meta[key], dict):
                sub_dict = list_of_dict_to_dict_of_list(values)
                for sub_key, sub_list in sub_dict.items():
                    try:
                        if np.isscalar(sub_list[0]):
                            sub_dict[sub_key] = np.array(sub_list).tolist()
                        else:
                            sub_dict[sub_key] = np.concatenate(sub_list, axis=0).tolist()
                    except Exception:
                        sub_dict[sub_key] = sub_list
                merged_meta[key] = sub_dict
            else:
                merged_meta[key] = values
        
        return LazyDataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=merged_meta)
    
    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1) -> "LazyDataProto":
        dp = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, num_batch_dims=num_batch_dims)
        return cls.from_dataproto(dp)
    
    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None) -> "LazyDataProto":
        dp = DataProto.from_single_dict(data=data, meta_info=meta_info)
        return cls.from_dataproto(dp)
