"""Step 13 all-cell geometry on the Step 15 typed-attention kernels.

The Rust builder ships only invariant edge fields and global indices. This
module derives the destination-, source-, and class-major CSR views on the
batch device, once per forward, then reuses :mod:`cell_latents` for the actual
attention and its deterministic reductions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .cell_latents import CellTables

ORBIT48_CLASSES = 48
RADIUS_CLASSES = ORBIT48_CLASSES * 2 * 2
ADJACENCY_CLASSES = 1
NEAREST_BUCKETS = 10
CELL_NODE_SCOPES = ("all", "uncovered")


def edge_tables(
    src: Tensor,
    dst: Tensor,
    cls: Tensor,
    covered_destinations: Tensor,
    n_sources: int,
    n_destinations: int,
    n_classes: int,
    uncovered_only: bool,
) -> CellTables:
    """Sort one typed edge family into the cell-attention's three views."""
    if not (src.shape == dst.shape == cls.shape) or src.ndim != 1:
        raise ValueError("src, dst, and cls must be one-dimensional and one length")
    if covered_destinations.ndim != 1:
        raise ValueError("covered_destinations must be one-dimensional")
    if src.numel():
        if int(src.min()) < 0 or int(src.max()) >= n_sources:
            raise ValueError(f"src entries must lie in [0, {n_sources})")
        if int(dst.min()) < 0 or int(dst.max()) >= n_destinations:
            raise ValueError(f"dst entries must lie in [0, {n_destinations})")
        if int(cls.min()) < 0 or int(cls.max()) >= n_classes:
            raise ValueError(f"cls entries must lie in [0, {n_classes})")
    if covered_destinations.numel() and (
        int(covered_destinations.min()) < 0
        or int(covered_destinations.max()) >= n_destinations
    ):
        raise ValueError(
            f"covered_destinations entries must lie in [0, {n_destinations})"
        )
    if uncovered_only:
        keep = ~torch.isin(dst, covered_destinations)
        src = src[keep]
        dst = dst[keep]
        cls = cls[keep]
    device = src.device

    order = torch.argsort(dst, stable=True)
    edge_dst = dst[order]
    edge_src = src[order]
    edge_class = cls[order]
    covered = torch.unique_consecutive(edge_dst)
    dst_ptr = torch.searchsorted(
        edge_dst, torch.arange(n_destinations + 1, device=device)
    )

    source_order = torch.argsort(edge_src, stable=True)
    source_ptr = torch.searchsorted(
        edge_src[source_order], torch.arange(n_sources + 1, device=device)
    )
    edge_sdst = edge_dst[source_order]
    edge_sclass = edge_class[source_order]
    edge_ssrc = edge_src[source_order]

    cedge_dst = torch.argsort(edge_class.to(torch.int32), stable=True)
    cedge_src = torch.argsort(edge_sclass.to(torch.int32), stable=True)
    cls_ptr = torch.searchsorted(
        edge_class[cedge_dst], torch.arange(n_classes + 1, device=device)
    )
    return CellTables(
        covered,
        dst_ptr,
        edge_src,
        edge_class,
        edge_dst,
        source_ptr,
        edge_sdst,
        edge_sclass,
        edge_ssrc,
        cls_ptr,
        cedge_dst,
        cedge_src,
    )


@torch.library.custom_op("mantisnet::cell_node_edge_tables", mutates_args=())
def derive_edge_tables(
    src: Tensor,
    dst: Tensor,
    cls: Tensor,
    covered_destinations: Tensor,
    n_sources: int,
    n_destinations: int,
    n_classes: int,
    uncovered_only: bool,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
]:
    """Opaque device-side edge-table derivation for compiled forwards."""
    return tuple(
        edge_tables(
            src,
            dst,
            cls,
            covered_destinations,
            n_sources,
            n_destinations,
            n_classes,
            uncovered_only,
        )
    )


@derive_edge_tables.register_fake
def _(
    src,
    dst,
    cls,
    covered_destinations,
    n_sources,
    n_destinations,
    n_classes,
    uncovered_only,
):
    ctx = torch.library.get_ctx()
    n_covered = ctx.new_dynamic_size()
    edges = ctx.new_dynamic_size() if uncovered_only else src.shape[0]

    def edge_rows():
        return src.new_empty((edges,), dtype=torch.long)

    return (
        src.new_empty((n_covered,), dtype=torch.long),
        src.new_empty((n_destinations + 1,), dtype=torch.long),
        edge_rows(), edge_rows(), edge_rows(),
        src.new_empty((n_sources + 1,), dtype=torch.long),
        edge_rows(), edge_rows(), edge_rows(),
        src.new_empty((n_classes + 1,), dtype=torch.long),
        edge_rows(), edge_rows(),
    )


def tables_from_op(
    src: Tensor,
    dst: Tensor,
    cls: Tensor,
    covered_destinations: Tensor,
    n_sources: int,
    n_destinations: int,
    n_classes: int,
    uncovered_only: bool,
) -> CellTables:
    return CellTables(
        *derive_edge_tables(
            src,
            dst,
            cls,
            covered_destinations,
            n_sources,
            n_destinations,
            n_classes,
            uncovered_only,
        )
    )
