"""
Test script for LazyDataProto with TransferQueue integration.
"""

import os
import sys

os.environ["ROLL_ENABLE_TQ"] = "1"

import torch
import numpy as np
from tensordict import TensorDict

from roll.distributed.scheduler import LazyDataProto, is_tq_enabled


def test_lazy_dataproto_basic():
    """Test basic LazyDataProto functionality without TransferQueue."""
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128, dtype=torch.long),
    }, batch_size=(4,))
    
    non_tensor_batch = {
        "domain": np.array(["math", "math", "code", "code"], dtype=object),
    }
    meta_info = {"step": 0, "epoch": 1}
    
    data = LazyDataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    
    assert len(data) == 4
    assert data.is_materialized
    assert not data.is_lazy
    print("Basic test passed")


def test_lazy_dataproto_pickle_mode():
    """Test pickle serialization when TQ is disabled."""
    os.environ["ROLL_ENABLE_TQ"] = "0"
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128, dtype=torch.long),
    }, batch_size=(4,))
    
    non_tensor_batch = {
        "domain": np.array(["math", "math", "code", "code"], dtype=object),
    }
    meta_info = {"step": 0}
    
    data = LazyDataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    state = data.__getstate__()
    assert state[0] == "pickle", f"Expected pickle mode, got {state[0]}"
    
    new_data = LazyDataProto.__new__(LazyDataProto)
    new_data.__setstate__(state)
    assert len(new_data) == 4
    assert new_data.is_materialized
    print("Pickle mode test passed")


def test_lazy_dataproto_tq_mode():
    """Test TransferQueue serialization when enabled."""
    os.environ["ROLL_ENABLE_TQ"] = "1"
    
    if not is_tq_enabled():
        print("TransferQueue not available, skipping TQ mode test")
        return
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128, dtype=torch.long),
    }, batch_size=(4,))
    
    non_tensor_batch = {
        "domain": np.array(["math", "math", "code", "code"], dtype=object),
    }
    meta_info = {"step": 0}
    
    data = LazyDataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    state = data.__getstate__()
    assert state[0] == "tq", f"Expected TQ mode, got {state[0]}"
    
    new_data = LazyDataProto.__new__(LazyDataProto)
    new_data.__setstate__(state)
    
    assert not new_data.is_materialized
    assert new_data.is_lazy
    assert len(new_data) == 4
    
    new_data._materialize()
    assert new_data.is_materialized
    assert not new_data.is_lazy
    assert "input_ids" in new_data.batch
    print("TransferQueue mode test passed")


def test_lazy_dataproto_from_tq_meta():
    """Test creating LazyDataProto from BatchMeta."""
    os.environ["ROLL_ENABLE_TQ"] = "1"
    
    if not is_tq_enabled():
        print("TransferQueue not available")
        return
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (8, 256)),
    }, batch_size=(8,))
    
    non_tensor_batch = {
        "uuid": np.array([f"id_{i}" for i in range(8)], dtype=object),
    }
    meta_info = {"global_step": 100}
    
    data = LazyDataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    
    meta = data.to_tq_meta(partition_id="test")
    assert meta is not None
    assert meta.size == 8
    
    lazy_data = LazyDataProto.from_tq_meta(meta)
    assert not lazy_data.is_materialized
    assert lazy_data.is_lazy
    assert len(lazy_data) == 8
    
    lazy_data._materialize()
    assert lazy_data.is_materialized
    assert "input_ids" in lazy_data.batch
    print("from_tq_meta test passed")


def test_lazy_dataproto_indexing():
    """Test indexing operations trigger materialization."""
    os.environ["ROLL_ENABLE_TQ"] = "1"
    
    if not is_tq_enabled():
        print("TransferQueue not available")
        return
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (8, 128)),
    }, batch_size=(8,))
    
    data = LazyDataProto(batch=batch)
    state = data.__getstate__()
    
    new_data = LazyDataProto.__new__(LazyDataProto)
    new_data.__setstate__(state)
    
    assert new_data.is_lazy
    
    item = new_data[0]
    assert new_data.is_materialized
    print("Indexing test passed")


def test_lazy_dataproto_slice():
    """Test slice operation."""
    os.environ["ROLL_ENABLE_TQ"] = "1"
    
    if not is_tq_enabled():
        print("TransferQueue not available")
        return
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (8, 128)),
    }, batch_size=(8,))
    
    data = LazyDataProto(batch=batch)
    sliced = data.slice(2, 6)
    
    assert len(sliced) == 4
    print("Slice test passed")


def test_lazy_dataproto_chunk():
    """Test chunk operation."""
    os.environ["ROLL_ENABLE_TQ"] = "1"
    
    if not is_tq_enabled():
        print("TransferQueue not available")
        return
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (8, 128)),
    }, batch_size=(8,))
    
    data = LazyDataProto(batch=batch)
    chunks = data.chunk(2)
    
    assert len(chunks) == 2
    assert len(chunks[0]) == 4
    assert len(chunks[1]) == 4
    print("Chunk test passed")


def test_lazy_dataproto_concat():
    """Test concat operation."""
    os.environ["ROLL_ENABLE_TQ"] = "1"
    
    if not is_tq_enabled():
        print("TransferQueue not available")
        return
    
    batch1 = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
    }, batch_size=(4,))
    
    batch2 = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
    }, batch_size=(4,))
    
    data1 = LazyDataProto(batch=batch1)
    data2 = LazyDataProto(batch=batch2)
    
    concatenated = LazyDataProto.concat([data1, data2])
    assert len(concatenated) == 8
    print("Concat test passed")


def test_lazy_dataproto_to_device():
    """Test to() device operation."""
    os.environ["ROLL_ENABLE_TQ"] = "1"
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
    }, batch_size=(4,))
    
    data = LazyDataProto(batch=batch)
    data_cpu = data.to("cpu")
    
    assert data_cpu.is_materialized
    print("To device test passed")


def test_lazy_dataproto_from_dict():
    """Test from_dict factory method."""
    tensors = {
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128, dtype=torch.long),
    }
    
    data = LazyDataProto.from_dict(tensors=tensors)
    assert len(data) == 4
    assert "input_ids" in data.batch
    print("from_dict test passed")


def test_lazy_dataproto_dataproto_conversion():
    """Test conversion between LazyDataProto and DataProto."""
    from roll.distributed.scheduler import DataProto
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
    }, batch_size=(4,))
    
    non_tensor_batch = {
        "domain": np.array(["math"] * 4, dtype=object),
    }
    meta_info = {"step": 42}
    
    dp = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    lazy_dp = LazyDataProto.from_dataproto(dp)
    
    assert len(lazy_dp) == 4
    assert lazy_dp.is_materialized
    
    dp_back = lazy_dp.to_dataproto()
    assert isinstance(dp_back, DataProto)
    assert len(dp_back) == 4
    assert dp_back.meta_info["step"] == 42
    print("DataProto conversion test passed")


if __name__ == "__main__":
    print("Testing LazyDataProto with TransferQueue integration\n")
    
    test_lazy_dataproto_basic()
    test_lazy_dataproto_pickle_mode()
    test_lazy_dataproto_tq_mode()
    test_lazy_dataproto_from_tq_meta()
    test_lazy_dataproto_indexing()
    test_lazy_dataproto_slice()
    test_lazy_dataproto_chunk()
    test_lazy_dataproto_concat()
    test_lazy_dataproto_to_device()
    test_lazy_dataproto_from_dict()
    test_lazy_dataproto_dataproto_conversion()
    
    print("\nAll tests passed!")
