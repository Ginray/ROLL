"""
Test script for TransferQueue integration with DataProto.
"""

import os
os.environ["ROLL_ENABLE_TQ"] = "1"

import torch
import numpy as np
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler import is_tq_enabled


def test_dataproto_pickle_mode():
    os.environ["ROLL_ENABLE_TQ"] = "0"
    
    batch = TensorDict({
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128, dtype=torch.long),
    }, batch_size=(4,))
    
    non_tensor_batch = {
        "domain": np.array(["math", "math", "code", "code"], dtype=object),
    }
    meta_info = {"step": 0}
    
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    state = data.__getstate__()
    assert state[0] == "pickle", f"Expected pickle mode, got {state[0]}"
    
    new_data = DataProto.__new__(DataProto)
    new_data.__setstate__(state)
    assert len(new_data) == 4
    print("Pickle mode test passed")


def test_dataproto_tq_mode():
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
    
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    state = data.__getstate__()
    assert state[0] == "tq", f"Expected TQ mode, got {state[0]}"
    
    new_data = DataProto.__new__(DataProto)
    new_data.__setstate__(state)
    assert len(new_data) == 4
    print("TransferQueue mode test passed")


def test_to_from_tq_meta():
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
    
    data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    
    meta = data.to_tq_meta(partition_id="test")
    assert meta is not None
    assert meta.size == 8
    
    restored = DataProto.from_tq_meta(meta)
    assert len(restored) == 8
    print("to_tq_meta/from_tq_meta test passed")


if __name__ == "__main__":
    print("Testing TransferQueue integration with DataProto\n")
    test_dataproto_pickle_mode()
    test_dataproto_tq_mode()
    test_to_from_tq_meta()
    print("\nAll tests passed!")
