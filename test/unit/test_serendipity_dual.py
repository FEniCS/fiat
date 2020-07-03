# Copyright (C) 2020 Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT import ufc_cell
from FIAT import Serendipity
import numpy as np
import pytest


@pytest.mark.parametrize("cell", ("quadrilateral", "hexahedron"))
@pytest.mark.parametrize("degree", range(1, 7))
def test_nodes(cell, degree):
    """Make sure that apply the nodes (via pt_dict) to the
    basis gives the Kronecker property"""
    T = ufc_cell(cell)
    z = tuple([0] * T.get_spatial_dimension())
    S = Serendipity(T, degree)

    nds = S.dual_basis()

    nSvals = np.zeros((len(nds), len(nds)))

    for i, nd in enumerate(nds):
        kpts = [(k, pt) for k, pt in enumerate(nd.pt_dict)]
        pts = [pt for (_, pt) in kpts]
        Svals = S.tabulate(0, pts)[z]

        # Node on each of the basis functions
        for j in range(len(nSvals)):  # loop over bfs
            # apply the node to the bfs
            for (k, pt) in kpts:
                wcs = nd.pt_dict[pt]
                for (w, _) in wcs:
                    nSvals[i, j] += w * Svals[j, k]

    assert np.allclose(nSvals, np.eye(*nSvals.shape))
