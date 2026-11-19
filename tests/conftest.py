"""Shared by every test module."""
import pytest


@pytest.fixture(scope="session", autouse=True)
def vector_math_started():
    """One tiny ``torch.exp`` before any test.

    torch 2.14.1+cu130 with MKL 2024.2, on the CPU: the first multi-threaded vector-math call of a
    process (``torch.exp``, ``torch.sin``, ...) is sometimes wrong by ~1e-4 in one thread's half of
    the elements, and after one small call every later one is right (measured in fresh processes:
    13 wrong of 570 without it, 0 of 530 with it). Without this, a test that compares two forward
    passes to better than 1e-4 fails in 2 to 10 % of the sessions when it holds the first forward
    pass (seen in test_glno_shapes_data.py, padded batch against each mesh alone). A script that
    makes such a comparison on the CPU needs the same call at its start.
    """
    try:
        import torch
    except ImportError:
        return
    torch.exp(torch.zeros(8))
