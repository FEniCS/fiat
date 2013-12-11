# Copyright (C) 2013 Andrew T. T. McRae
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2013-11-03
# Last changed: 2013-11-12

import sys, numpy, FIAT

def test():
    """Tests for lack-of-exception rather than correctness"""
    ### DEFINE REFERENCE ELEMENTS
    S = FIAT.reference_element.UFCTriangle()
    T = FIAT.reference_element.UFCInterval()

    ### DEFINE 1D FUNCTION SPACES
    P0_T = FIAT.DiscontinuousLagrange(T, 0)
    P1DG_T = FIAT.DiscontinuousLagrange(T, 1)
    P2DG_T = FIAT.DiscontinuousLagrange(T, 2)
    P1_T = FIAT.Lagrange(T, 1)
    P2_T = FIAT.Lagrange(T, 2)
    list1d = [P0_T, P1DG_T, P2DG_T, P1_T, P2_T]

    ### INTERVAL x INTERVAL tests ###
    for A in list1d:
        for B in list1d:
            C = FIAT.TensorFiniteElement(A, B)
            assert C.value_shape() == ()
            C.tabulate(1, [(0.1, 0.2), (0.3, 0.4)])
    # Keep some interesting combinations for later
    P1P1_TT = FIAT.TensorFiniteElement(P1_T, P1_T)
    P0P0_TT = FIAT.TensorFiniteElement(P0_T, P0_T)
    P0P1_TT = FIAT.TensorFiniteElement(P0_T, P1_T)
    P1P0_TT = FIAT.TensorFiniteElement(P1_T, P0_T)
    
    # These should all fail
    #FIAT.Hdiv(P1P1_TT)
    #FIAT.Hcurl(P1P1_TT)
    #FIAT.Hdiv(P0P0_TT)
    #FIAT.Hcurl(P0P0_TT)

    # These should all work [n-1 and 1 are identical]
    P0P1_TTdiv = FIAT.Hdiv(P0P1_TT)
    P0P1_TTcurl = FIAT.Hcurl(P0P1_TT)
    P1P0_TTdiv = FIAT.Hdiv(P1P0_TT)
    P1P0_TTcurl = FIAT.Hcurl(P1P0_TT)
    RT_square = FIAT.EnrichedElement(P0P1_TTdiv, P1P0_TTdiv)
    Ned_square = FIAT.EnrichedElement(P0P1_TTcurl, P1P0_TTcurl)
    for C in [P0P1_TTdiv, P0P1_TTcurl, P1P0_TTdiv, P1P0_TTcurl, RT_square, Ned_square]:
        assert C.value_shape() == (2,)
        C.tabulate(1, [(0.1, 0.2), (0.3, 0.4)])
    
    ### DEFINE 2D FUNCTION SPACES
    P0_S = FIAT.DiscontinuousLagrange(S, 0)
    P1DG_S = FIAT.DiscontinuousLagrange(S, 1)
    P2DG_S = FIAT.DiscontinuousLagrange(S, 2)
    P3DG_S = FIAT.DiscontinuousLagrange(S, 3)
    P1_S = FIAT.Lagrange(S, 1)
    P2_S = FIAT.Lagrange(S, 2)
    P3_S = FIAT.Lagrange(S, 3)
    BDM1_S = FIAT.BrezziDouglasMarini(S, 1)
    BDM2_S = FIAT.BrezziDouglasMarini(S, 2)
    BDM3_S = FIAT.BrezziDouglasMarini(S, 3)
    BDFM2_S = FIAT.BrezziDouglasFortinMarini(S, 2)
    RT1_S = FIAT.RaviartThomas(S, 1)
    RT2_S = FIAT.RaviartThomas(S, 2)
    RT3_S = FIAT.RaviartThomas(S, 3)
    Ned1_S = FIAT.Nedelec(S, 1)
    Ned2_S = FIAT.Nedelec(S, 2)
    Ned3_S = FIAT.Nedelec(S, 3)
    NedS1_S = FIAT.NedelecSecondKind(S, 1)
    NedS2_S = FIAT.NedelecSecondKind(S, 2)
    NedS3_S = FIAT.NedelecSecondKind(S, 3)
    
    list2d_scalar = [P0_S, P1DG_S, P2DG_S, P3DG_S, P1_S, P2_S, P3_S]
    list2d_hdiv = [RT1_S, RT2_S, RT3_S, BDFM2_S, BDM1_S, BDM2_S, BDM3_S]
    list2d_hcurl = [Ned1_S, NedS1_S, Ned2_S, Ned3_S, NedS2_S, NedS3_S]

    ### TRIANGLE x INTERVAL tests ###
    for A in list2d_scalar:
        for B in list1d:
            C = FIAT.TensorFiniteElement(A, B)
            assert C.value_shape() == ()
            C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])

    for A in list2d_hdiv:
        for B in list1d:
            C = FIAT.TensorFiniteElement(A, B)
            assert C.value_shape() == (2,)
            C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])

    for A in list2d_hcurl:
        for B in list1d:
            C = FIAT.TensorFiniteElement(A, B)
            assert C.value_shape() ==  (2,)
            C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])

    # Keep some interesting combinations for later
    P2P1_ST = FIAT.TensorFiniteElement(P2_S, P1_T)
    P0P0_ST = FIAT.TensorFiniteElement(P0_S, P0_T)
    
    BDM1P0_ST = FIAT.TensorFiniteElement(BDM1_S, P0_T)
    P0P1_ST = FIAT.TensorFiniteElement(P0_S, P1_T)

    BDM1P1_ST = FIAT.TensorFiniteElement(BDM1_S, P1_T)
    P2P0_ST = FIAT.TensorFiniteElement(P2_S, P0_T)

    # These should all fail
    #FIAT.Hdiv(P2P1_ST)
    #FIAT.Hcurl(P2P1_ST)
    #FIAT.Hdiv(P0P0_ST)
    #FIAT.Hcurl(P0P0_ST)

    # Hdiv only works on (n-1)-forms, i.e. 2-forms
    # Hcurl only works on 1-forms

    BDM1P0_STdiv = FIAT.Hdiv(BDM1P0_ST)
    #FIAT.Hcurl(BDM1P0_ST)
    
    P0P1_STdiv = FIAT.Hdiv(P0P1_ST)
    #FIAT.Hcurl(P0P1_ST)
    
    BDM1P1_STcurl = FIAT.Hcurl(BDM1P1_ST)
    #FIAT.Hdiv(BDM1P1_ST)
    
    P2P0_STcurl = FIAT.Hcurl(P2P0_ST)
    #FIAT.Hdiv(P2P0_ST)
    
    BDMwedge = FIAT.EnrichedElement(BDM1P0_STdiv, P0P1_STdiv)
    Nedwedge = FIAT.EnrichedElement(BDM1P1_STcurl, P2P0_STcurl)

    for C in [BDM1P0_STdiv, P0P1_STdiv, BDM1P1_STcurl, P2P0_STcurl, BDMwedge, Nedwedge]:
        assert C.value_shape() == (3,)
        C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])
    
    ### (INTERVAL x INTERVAL) x INTERVAL tests ###
    P0P0P0_TTT = FIAT.TensorFiniteElement(P0P0_TT, P0_T)
    
    P0P0P1_TTT = FIAT.TensorFiniteElement(P0P0_TT, P1_T)
    P0P1P0_TTT_a = FIAT.TensorFiniteElement(P0P1_TTdiv, P0_T)
    P0P1P0_TTT_b = FIAT.TensorFiniteElement(P0P1_TTcurl, P0_T)
    P1P0P0_TTT_a = FIAT.TensorFiniteElement(P1P0_TTdiv, P0_T)
    P1P0P0_TTT_b = FIAT.TensorFiniteElement(P1P0_TTcurl, P0_T)

    P0P1P1_TTT_a = FIAT.TensorFiniteElement(P0P1_TTcurl, P1_T)
    P0P1P1_TTT_b = FIAT.TensorFiniteElement(P0P1_TTdiv, P1_T)
    P1P0P1_TTT_a = FIAT.TensorFiniteElement(P1P0_TTcurl, P1_T)
    P1P0P1_TTT_b = FIAT.TensorFiniteElement(P1P0_TTdiv, P1_T)
    P1P1P0_TTT = FIAT.TensorFiniteElement(P1P1_TT, P0_T)

    P1P1P1_TTT = FIAT.TensorFiniteElement(P1P1_TT, P1_T)
    
    for C in [P0P0P0_TTT, P1P1P1_TTT]:
        # scalar x scalar, remains affine
        assert C.value_shape() == ()
        C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])
    
    for C in [P0P0P1_TTT, P1P1P0_TTT]:
        # scalar x scalar, but will Hdiv/Hcurl later
        assert C.value_shape() == ()
        C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])
    
    for C in [P0P1P0_TTT_a, P0P1P0_TTT_b, P1P0P0_TTT_a, P1P0P0_TTT_b]:
        assert C.value_shape() == (2,)
        C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])

    for C in [P0P1P1_TTT_a, P0P1P1_TTT_b, P1P0P1_TTT_a, P1P0P1_TTT_b]:
        assert C.value_shape() == (2,)
        C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])
    
    P0P0P1_TTTdiv = FIAT.Hdiv(P0P0P1_TTT)
    #FIAT.Hcurl(P0P0P1_TTT)
    
    P0P1P0_TTTdiv_a = FIAT.Hdiv(P0P1P0_TTT_a)
    #FIAT.Hcurl(P0P1P0_TTTa)
    P0P1P0_TTTdiv_b = FIAT.Hdiv(P0P1P0_TTT_b)
    #FIAT.Hcurl(P0P1P0_TTTb)
    
    # Did you spot the magic? If not, look carefully at how these spaces were made!
    assert numpy.all(P0P1P0_TTTdiv_a.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)] == P0P1P0_TTTdiv_b.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)])

    P1P0P0_TTTdiv_a = FIAT.Hdiv(P1P0P0_TTT_a)
    #FIAT.Hcurl(P1P0P0_TTT_a)

    P1P0P0_TTTdiv_b = FIAT.Hdiv(P1P0P0_TTT_b)
    #FIAT.Hcurl(P1P0P0_TTT_b)
    
    # More magic
    assert numpy.all(P1P0P0_TTTdiv_a.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)] == P1P0P0_TTTdiv_b.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)])

    P0P1P1_TTTcurl_a = FIAT.Hcurl(P0P1P1_TTT_a)
    #FIAT.Hdiv(P0P1P1_TTT_a)

    P0P1P1_TTTcurl_b = FIAT.Hcurl(P0P1P1_TTT_b)
    #FIAT.Hdiv(P0P1P1_TTT_b)

    # More magic
    assert numpy.all(P0P1P1_TTTcurl_a.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)] == P0P1P1_TTTcurl_b.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)])

    P1P0P1_TTTcurl_a = FIAT.Hcurl(P1P0P1_TTT_a)
    #FIAT.Hdiv(P0P1P1_TTT_a)

    P1P0P1_TTTcurl_b = FIAT.Hcurl(P1P0P1_TTT_b)
    #FIAT.Hdiv(P0P1P1_TTT_b)

    # More magic
    assert numpy.all(P1P0P1_TTTcurl_a.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)] == P1P0P1_TTTcurl_b.tabulate(0, [(0.1, 0.1, 0.1)])[(0,0,0)])
    
    P1P1P0_TTTcurl = FIAT.Hcurl(P1P1P0_TTT)
    
    RT_cube = FIAT.EnrichedElement(FIAT.EnrichedElement(P0P0P1_TTTdiv, P0P1P0_TTTdiv_a), P1P0P0_TTTdiv_a)
    
    Ned_cube = FIAT.EnrichedElement(FIAT.EnrichedElement(P0P1P1_TTTcurl_a, P1P0P1_TTTcurl_a), P1P1P0_TTTcurl)
    
    for C in [P0P0P1_TTTdiv, P0P1P0_TTTdiv_a, P0P1P0_TTTdiv_b, P1P0P0_TTTdiv_a, P1P0P0_TTTdiv_b, P0P1P1_TTTcurl_a, P0P1P1_TTTcurl_b, P1P0P1_TTTcurl_a, P1P0P1_TTTcurl_b, P1P1P0_TTTcurl, RT_cube, Ned_cube]:
        assert C.value_shape() == (3,)
        C.tabulate(1, [(0.1, 0.2, 0.3), (0.3, 0.4, 0.5)])

    return 0

if __name__ == "__main__":
    sys.exit(test())
