"""Process-wide persistent multiprocessing pool used during inference.

Forking a fresh `multiprocessing.Pool` for every call is expensive on the order of tens to
hundreds of milliseconds (mostly fork() + IPC startup + module re-imports in workers). Reusing a
single pool across calls amortizes this cost.
"""
from __future__ import annotations

import atexit
import multiprocessing
import os
from typing import Optional

_pool: Optional[multiprocessing.pool.Pool] = None
_pool_size: int = 0


def get_pool(num_workers: int) -> Optional[multiprocessing.pool.Pool]:
    """Return the singleton pool, lazily creating it. Returns ``None`` when disabled."""
    global _pool, _pool_size

    if os.environ.get("RC_DISABLE_POOL", "0") == "1":
        return None

    if _pool is None or _pool_size != num_workers:
        if _pool is not None:
            _pool.close()
            _pool.join()
        _pool = multiprocessing.Pool(num_workers)
        _pool_size = num_workers
        atexit.register(_shutdown)
    return _pool


def _shutdown() -> None:
    global _pool
    if _pool is not None:
        try:
            _pool.close()
            _pool.join()
        except Exception:
            pass
        _pool = None
