"""
TransferQueue Integration Example for ROLL

This example demonstrates how to use TransferQueue with ROLL for efficient
data transfer in distributed training workflows.

TransferQueue is a high-performance data storage and transfer module optimized
for post-training workflows. It provides:
- Fine-grained data management at sub-sample level
- Load balancing capabilities
- Streaming scheduling
- Decoupling of data dependencies across computational tasks

Installation:
    pip install transfer_queue

Enable TransferQueue:
    export ROLL_ENABLE_TQ=1

Usage Example:
"""

import os

os.environ["ROLL_ENABLE_TQ"] = "1"

from roll.distributed.scheduler import (
    register,
    Dispatch,
    DataProto,
    is_tq_available,
    is_tq_enabled,
    init_transfer_queue,
    TransferQueueManager,
    tqbridge,
)
from roll.distributed.scheduler.protocol import TensorDict
import torch


def example_basic_usage():
    """Basic TransferQueue usage example."""
    print("=" * 60)
    print("TransferQueue Basic Usage Example")
    print("=" * 60)

    if not is_tq_available():
        print("TransferQueue is not installed. Install with: pip install transfer_queue")
        return

    print(f"TransferQueue available: {is_tq_available()}")
    print(f"TransferQueue enabled: {is_tq_enabled()}")

    if is_tq_enabled():
        init_transfer_queue()
        print("TransferQueue initialized successfully")


def example_with_decorator():
    """Example using the register decorator with TransferQueue."""

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, enable_tq=True)
    def process_data(data: DataProto) -> DataProto:
        """
        Process data with TransferQueue support.

        When enable_tq=True and TransferQueue is available:
        - Input BatchMeta is automatically converted to DataProto
        - Output DataProto is automatically converted back to BatchMeta
        """
        result = DataProto(
            batch=TensorDict(
                {"output": data.batch["input"] * 2},
                batch_size=data.batch.batch_size,
            ),
            meta_info=data.meta_info,
        )
        return result

    print("Decorator example defined successfully")


def example_with_tqbridge():
    """Example using the tqbridge decorator directly."""

    @tqbridge(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    async def async_process_data(data: DataProto) -> DataProto:
        """
        Async process data with TransferQueue support.

        The tqbridge decorator handles:
        - BatchMeta -> DataProto conversion on input
        - DataProto -> BatchMeta conversion on output
        """
        result = DataProto(
            batch=TensorDict(
                {"processed": torch.randn(data.batch.batch_size[0], 128)},
                batch_size=data.batch.batch_size,
            ),
            meta_info=data.meta_info,
        )
        return result

    print("tqbridge example defined successfully")


def example_transfer_queue_manager():
    """Example using TransferQueueManager directly."""

    if not is_tq_available():
        print("TransferQueue not available, skipping manager example")
        return

    manager = TransferQueueManager(partition_id="train_partition")
    manager.init()

    data = DataProto(
        batch=TensorDict(
            {"input": torch.randn(32, 128)},
            batch_size=(32,),
        ),
        meta_info={"epoch": 1},
    )

    meta = manager.put_data(data, partition_id="train_partition")
    print(f"Data stored with meta size: {meta.size}")

    retrieved_data = manager.get_data(meta)
    print(f"Data retrieved with batch size: {len(retrieved_data)}")


def example_kv_operations():
    """Example using KV operations for key-value based data retrieval."""
    from roll.distributed.scheduler import kv_batch_meta_to_batch_meta, batch_meta_to_kv_batch_meta

    if not is_tq_available():
        print("TransferQueue not available, skipping KV operations example")
        return

    print("KV operations example - functions available:")
    print("  - kv_batch_meta_to_batch_meta: Convert KVBatchMeta to BatchMeta")
    print("  - batch_meta_to_kv_batch_meta: Convert BatchMeta to KVBatchMeta")


if __name__ == "__main__":
    print("\nTransferQueue Integration Examples\n")

    example_basic_usage()
    print()

    example_with_decorator()
    print()

    example_with_tqbridge()
    print()

    example_transfer_queue_manager()
    print()

    example_kv_operations()
    print()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
