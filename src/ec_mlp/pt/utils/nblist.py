# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Optional, Tuple

import torch
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
from vesin.torch import NeighborList


def dp_nblist(
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    nnei: int,
    rcut: float,
):
    """
    Build neighbor list data based on DP functions.
    """
    positions = torch.reshape(positions, [1, -1, 3])
    (
        extended_coord,
        extended_atype,
        mapping,
        nlist,
    ) = extend_input_and_build_neighbor_list(
        positions,
        torch.zeros(1, positions.shape[1]),
        rcut,
        [nnei],
        box=box,
    )
    extended_pairs = make_extended_pairs(nlist)
    pairs, buffer_scales, mask_ij, mask_ii = make_local_pairs(extended_pairs, mapping)
    ds_ij = make_ds(extended_pairs, extended_coord, mask_ij)
    ds_ii = make_ds(extended_pairs, extended_coord, mask_ii)
    ds = torch.concat([ds_ij, ds_ii])
    del extended_coord, extended_atype
    return pairs, ds, buffer_scales


def vesin_nblist(
    positions: torch.Tensor,
    box: Optional[torch.Tensor],
    rcut: float,
):
    device = positions.device
    calculator = NeighborList(cutoff=rcut, full_list=False)
    ii, jj, ds = calculator.compute(
        points=positions.to("cpu"),
        box=box.to("cpu"),
        periodic=True,
        quantities="ijd",
    )
    buffer_scales = torch.ones_like(ds).to(device)
    return torch.stack([ii, jj]).to(device).T, ds.to(device), buffer_scales


def make_extended_pairs(
    nlist: torch.Tensor,
) -> torch.Tensor:
    """Return the pairs between local and extended indices.

    Parameters
    ----------
    nlist : torch.Tensor
        nframes x nloc x nsel, neighbor list between local and extended indices

    Returns
    -------
    extended_pairs: torch.Tensor
        [[i1, j1], [i2, j2], ...],
        in which i is the local index and j is the extended index
    """
    nframes, nloc, nsel = nlist.shape
    assert nframes == 1
    nlist_reshape = torch.reshape(nlist, [nframes, nloc * nsel, 1])
    # nlist is padded with -1
    mask = nlist_reshape.ge(0)

    ii = torch.arange(nloc, dtype=torch.int64, device=nlist.device)
    ii = torch.tile(ii.reshape(-1, 1), [1, nsel])
    ii = torch.reshape(ii, [nframes, nloc * nsel, 1])
    sel_ii = torch.masked_select(ii, mask)

    # nf x (nloc x nsel)
    sel_jj = torch.masked_select(nlist_reshape, mask)
    extended_pairs = torch.stack([sel_ii, sel_jj], dim=-1)
    return extended_pairs


def make_local_pairs(
    extended_pairs: torch.Tensor,
    mapping: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the pairs between local indices.

    Parameters
    ----------
    extended_pairs : torch.Tensor
        npairs_all x 2,
    mapping : torch.Tensor
        nframes x nall, index from extended to local

    Returns
    -------
    local_pairs: torch.Tensor
        npairs_loc x 2, [[i1, j1], [i2, j2], ...],
        in which i and j are the local indices of the atoms (i < j)
    mask: torch.Tensor
        npairs_all, mask for the local pairs (i < j)
    """
    nframes, _nall = mapping.shape
    assert nframes == 1
    ii = extended_pairs[..., 0]
    jj = torch.gather(mapping.reshape(-1), 0, extended_pairs[..., 1])

    mask_ij = ii.lt(jj)
    mask_ii = ii.eq(jj)
    local_pairs_ij = torch.stack([ii, jj], dim=-1)[mask_ij]
    local_pairs_ii = torch.stack([ii, jj], dim=-1)[mask_ii]

    buffer_scales_ij = torch.ones(local_pairs_ij.shape[0], dtype=torch.float64)
    buffer_scales_ii = torch.ones(local_pairs_ii.shape[0], dtype=torch.float64) / 2.0

    local_pairs = torch.concat([local_pairs_ij, local_pairs_ii])
    buffer_scales = torch.concat([buffer_scales_ij, buffer_scales_ii])
    return local_pairs, buffer_scales, mask_ij, mask_ii


def make_ds(
    extended_pairs: torch.Tensor,
    extended_coord: torch.Tensor,
    pairs_mask: torch.Tensor,
) -> torch.Tensor:
    """Calculate the i-j distance from the neighbor list.

    Parameters
    ----------
    extended_pairs : torch.Tensor
        npairs_all x 2,
    extended_coord : torch.Tensor
        nframes x nall x 3, extended coordinates
    pairs_mask : torch.Tensor
        npairs_all, mask for the local pairs (i < j)

    Returns
    -------
    ds: torch.Tensor
        npairs_loc, i-j distance
    """
    nframes, _nall, _ = extended_coord.shape
    assert nframes == 1

    ii = extended_pairs[..., 0]
    jj = extended_pairs[..., 1]
    diff = extended_coord[:, jj] - extended_coord[:, ii]
    ds = torch.norm(diff.reshape(-1, 3)[pairs_mask], dim=-1)
    return ds


def sort_pairs(pairs: torch.Tensor) -> torch.Tensor:
    """
    Sort pairs first by the first index, then by the second index.
    """
    indices = torch.argsort(pairs[:, 1])
    pairs = pairs[indices]
    indices = torch.argsort(pairs[:, 0], stable=True)
    sorted_pairs = pairs[indices]
    return sorted_pairs
