# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TransferQueue adapter for ROLL.
This module provides integration with TransferQueue for efficient data transfer
in distributed training workflows.

Ref: https://github.com/verl-project/verl/pull/5401
TransferQueue: https://github.com/Ascend/TransferQueue
"""

import asyncio
import copy
import functools
import inspect
import logging
import os
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

if TYPE_CHECKING:
    from roll.distributed.scheduler.decorator import Dispatch

logger = get_logger()

TQ_AVAILABLE = False
TQ_INITIALIZED = False

try:
    import transfer_queue as tq
    from transfer_queue import BatchMeta, KVBatchMeta
    TQ_AVAILABLE = True
except ImportError:
    logger.warning("transfer_queue is not installed. TransferQueue features will be disabled.")
    BatchMeta = None
    KVBatchMeta = None


def is_tq_available() -> bool:
    return TQ_AVAILABLE


def is_tq_enabled() -> bool:
    return TQ_AVAILABLE and os.getenv("ROLL_ENABLE_TQ", "0") == "1"


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="tq-async-runner",
        daemon=True,
    )

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()


def _find_meta(*args, **kwargs) -> Optional["BatchMeta"]:
    if not TQ_AVAILABLE:
        return None
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta):
            return v
    return None


def _extract_non_tensor_data(data: DataProto) -> Dict[str, Any]:
    non_tensor_data = {}
    if data.non_tensor_batch:
        for k, v in data.non_tensor_batch.items():
            non_tensor_data[k] = v
    if data.meta_info:
        non_tensor_data["meta_info"] = copy.deepcopy(data.meta_info)
    return non_tensor_data


def _assign_non_tensor_data(data: DataProto, non_tensor_data: Dict[str, Any]) -> None:
    for k, v in non_tensor_data.items():
        if k == "meta_info":
            data.meta_info.update(v)
        else:
            data.non_tensor_batch[k] = v


async def _async_meta_to_dataproto(meta: "BatchMeta") -> DataProto:
    meta_info = copy.deepcopy(meta.extra_info)

    if meta.size == 0:
        empty_proto = DataProto(batch=TensorDict({}, batch_size=(0,)), meta_info=meta_info)
        return empty_proto

    tq_client = tq.get_client()
    tensordict = await tq_client.async_get_data(meta)

    non_tensor_batch = {}
    batch_size = tensordict.batch_size[0] if tensordict.batch_size else 0
    
    for key, val in meta_info.items():
        if isinstance(val, (NonTensorData, NonTensorStack)):
            non_tensor_batch[key] = np.array([val] * batch_size, dtype=object)
        elif isinstance(val, dict) and "non_tensor_batch" in val:
            for k, v in val["non_tensor_batch"].items():
                if isinstance(v, np.ndarray) and v.dtype == object:
                    non_tensor_batch[k] = v
                else:
                    non_tensor_batch[k] = np.array([v] * batch_size, dtype=object)
        elif isinstance(val, dict) and "meta_info" in val:
            pass
        else:
            non_tensor_batch[key] = np.array([val] * batch_size, dtype=object)

    return DataProto(batch=tensordict, non_tensor_batch=non_tensor_batch, meta_info=meta_info.get("meta_info", {}))


def _meta_to_dataproto(meta: "BatchMeta") -> DataProto:
    return _run_async_in_temp_loop(_async_meta_to_dataproto, meta)


async def _async_dataproto_to_meta(data: DataProto, meta: "BatchMeta", func_name: str = None) -> "BatchMeta":
    if data.batch is None or len(data) == 0:
        return meta

    fields = list(data.batch.keys())
    meta_data = {}

    if data.non_tensor_batch:
        meta_data["non_tensor_batch"] = copy.deepcopy(data.non_tensor_batch)
    if data.meta_info:
        meta_data["meta_info"] = copy.deepcopy(data.meta_info)

    if fields:
        t1 = time.time()
        tq_client = tq.get_client()
        meta = await tq_client.async_put(data=data.batch, metadata=meta)
        t2 = time.time()
        logger.debug(f"Task {func_name} wrote to TransferQueue, cost: {t2 - t1:.3f}s")

    meta.extra_info = meta_data
    return meta


def _dataproto_to_meta(data: DataProto, meta: "BatchMeta", func_name: str = None) -> "BatchMeta":
    return _run_async_in_temp_loop(_async_dataproto_to_meta, data, meta, func_name)


def _compute_need_collect(dispatch_mode: Union["Dispatch", Dict], args: list) -> bool:
    from roll.distributed.scheduler.decorator import Dispatch

    if dispatch_mode is None or isinstance(dispatch_mode, Dispatch):
        return True

    if isinstance(dispatch_mode, dict) and "collect_fn" in dispatch_mode:
        return True

    return True


def _postprocess_output(output, put_data: bool, need_collect: bool):
    if put_data and not need_collect:
        if TQ_AVAILABLE:
            return BatchMeta()
        return DataProto()

    if not put_data and not need_collect:
        if isinstance(output, DataProto):
            return DataProto()
        elif isinstance(output, TensorDict):
            return TensorDict({}, batch_size=(0,))

    return output


async def _async_kv_batch_meta_to_batch_meta(meta: "KVBatchMeta") -> "BatchMeta":
    global TQ_INITIALIZED

    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True

    tq_client = tq.get_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(
        keys=meta.keys, partition_id=meta.partition_id, create=False
    )

    fields = meta.fields
    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)

    batch_meta.extra_info = meta.extra_info
    return batch_meta


def kv_batch_meta_to_batch_meta(meta: "KVBatchMeta") -> "BatchMeta":
    return _run_async_in_temp_loop(_async_kv_batch_meta_to_batch_meta, meta)


async def _async_batch_meta_to_kv_batch_meta(meta: "BatchMeta") -> "KVBatchMeta":
    global TQ_INITIALIZED

    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True

    tq_client = tq.get_client()
    partition_id = meta.partition_ids[0]

    assert all([partition_id == pid for pid in meta.partition_ids]), \
        "All partition IDs must be the same"

    keys = await tq_client.async_kv_retrieve_keys(
        global_indexes=meta.global_indexes, partition_id=partition_id
    )

    kv_batch_meta = KVBatchMeta(
        keys=keys,
        tags=[{}] * meta.size,
        partition_id=partition_id,
        fields=meta.field_names,
        extra_info=meta.extra_info,
    )
    return kv_batch_meta


def batch_meta_to_kv_batch_meta(meta: "BatchMeta") -> "KVBatchMeta":
    return _run_async_in_temp_loop(_async_batch_meta_to_kv_batch_meta, meta)


def tqbridge(dispatch_mode: Union["Dispatch", Dict] = None):
    """
    Decorator for bridging TransferQueue BatchMeta and DataProto.

    This decorator automatically handles conversions between `BatchMeta`
    and `DataProto` in function parameters, and decides whether to sync
    function output back to `BatchMeta` based on configuration.

    Args:
        dispatch_mode: Controls data collection behavior for the current worker.

    Returns:
        A decorator function used to decorate target functions.
    """
    if not TQ_AVAILABLE:
        def noop_decorator(func):
            return func
        return noop_decorator

    from roll.distributed.scheduler.decorator import _check_dispatch_mode
    _check_dispatch_mode(dispatch_mode)

    def decorator(func):
        pid = os.getpid()

        @wraps(func)
        def inner(*args, **kwargs):
            batch_meta = _find_meta(*args, **kwargs)

            if batch_meta is None:
                return func(*args, **kwargs)

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            t1 = time.time()
            args = [_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
            kwargs = {k: _meta_to_dataproto(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()}
            t2 = time.time()

            logger.debug(f"Task {func.__name__} (pid={pid}) got {batch_meta.size} samples, cost: {t2 - t1:.3f}s")

            output = func(*args, **kwargs)

            put_data = False
            if isinstance(output, DataProto):
                if output.batch is not None and len(output) > 0:
                    assert len(output) == batch_meta.size, \
                        f"output batch size {len(output)} != meta size {batch_meta.size}"
                    put_data = True

            need_collect = _compute_need_collect(dispatch_mode, args)

            if put_data and need_collect:
                updated_meta = _dataproto_to_meta(output, batch_meta, func.__name__)
                return updated_meta

            return _postprocess_output(output, put_data, need_collect)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            batch_meta = _find_meta(*args, **kwargs)

            if batch_meta is None:
                return await func(*args, **kwargs)

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            t1 = time.time()
            args = [await _async_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
            kwargs = {
                k: await _async_meta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }
            t2 = time.time()

            logger.debug(f"Task {func.__name__} (pid={pid}) got {batch_meta.size} samples, cost: {t2 - t1:.3f}s")

            output = await func(*args, **kwargs)

            put_data = False
            if isinstance(output, DataProto):
                if output.batch is not None and len(output) > 0:
                    assert len(output) == batch_meta.size, \
                        f"output batch size {len(output)} != meta size {batch_meta.size}"
                    put_data = True

            need_collect = _compute_need_collect(dispatch_mode, args)

            if put_data and need_collect:
                updated_meta = await _async_dataproto_to_meta(output, batch_meta, func.__name__)
                return updated_meta

            return _postprocess_output(output, put_data, need_collect)

        if inspect.iscoroutinefunction(func):
            return async_inner
        return inner

    return decorator


class TransferQueueManager:
    """
    Manager class for TransferQueue operations.
    Provides high-level APIs for data transfer in distributed training.
    """

    def __init__(self, partition_id: Optional[str] = None):
        self.partition_id = partition_id
        self._initialized = False

    def init(self):
        if not TQ_AVAILABLE:
            raise RuntimeError("transfer_queue is not installed")

        global TQ_INITIALIZED
        if not TQ_INITIALIZED:
            tq.init()
            TQ_INITIALIZED = True
        self._initialized = True

    def put_data(self, data: DataProto, partition_id: Optional[str] = None) -> "BatchMeta":
        if not self._initialized:
            self.init()

        partition_id = partition_id or self.partition_id
        if partition_id is None:
            raise ValueError("partition_id is required")

        tq_client = tq.get_client()
        batch_size = len(data)
        
        meta = _run_async_in_temp_loop(
            tq_client.async_put,
            data.batch,
            BatchMeta(
                global_indexes=list(range(batch_size)),
                partition_ids=[partition_id] * batch_size,
            )
        )

        meta.extra_info = _extract_non_tensor_data(data)
        return meta

    def get_data(self, meta: "BatchMeta") -> DataProto:
        if not self._initialized:
            self.init()

        return _meta_to_dataproto(meta)

    def clear_data(self, meta: "BatchMeta"):
        if not self._initialized:
            self.init()

        tq_client = tq.get_client()
        _run_async_in_temp_loop(tq_client.async_clear_data, meta)


def init_transfer_queue():
    global TQ_INITIALIZED

    if not TQ_AVAILABLE:
        logger.warning("transfer_queue is not installed, skipping initialization")
        return False

    if not TQ_INITIALIZED:
        import sys
        tq_module_path = os.path.dirname(tq.__file__)
        if tq_module_path not in sys.path:
            sys.path.insert(0, tq_module_path)
        
        tq.init()
        TQ_INITIALIZED = True
        logger.info("TransferQueue initialized successfully")

    return True
