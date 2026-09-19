#!/usr/bin/env python3
"""Deprecated location; use ``python -m equicdft.data.mean_density`` or the
``equicdft-mean-density`` console script."""

from equicdft.data.mean_density import compute_mean_densities, main

__all__ = ["compute_mean_densities", "main"]

if __name__ == "__main__":
    main()
