"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

from __future__ import absolute_import, print_function, division

import pkg_resources

# Import finite element classes
from FIAT.finite_element import FiniteElement, CiarletElement  # noqa: F401
from FIAT.argyris import Argyris
from FIAT.argyris import QuinticArgyris
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.brezzi_douglas_fortin_marini import BrezziDouglasFortinMarini
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.discontinuous_taylor import DiscontinuousTaylor
from FIAT.discontinuous_raviart_thomas import DiscontinuousRaviartThomas
from FIAT.hermite import CubicHermite
from FIAT.lagrange import Lagrange
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
from FIAT.gauss_legendre import GaussLegendre
from FIAT.morley import Morley
from FIAT.nedelec import Nedelec
from FIAT.nedelec_second_kind import NedelecSecondKind
from FIAT.P0 import P0
from FIAT.raviart_thomas import RaviartThomas
from FIAT.crouzeix_raviart import CrouzeixRaviart
from FIAT.regge import Regge
from FIAT.hellan_herrmann_johnson import HellanHerrmannJohnson
from FIAT.bubble import Bubble
from FIAT.tensor_product import TensorProductElement
from FIAT.enriched import EnrichedElement
from FIAT.nodal_enriched import NodalEnrichedElement
from FIAT.discontinuous import DiscontinuousElement
from FIAT.hdiv_trace import HDivTrace
from FIAT.restricted import RestrictedElement             # noqa: F401

# Important functionality
from FIAT.quadrature import make_quadrature               # noqa: F401
from FIAT.quadrature_schemes import create_quadrature     # noqa: F401
from FIAT.reference_element import ufc_cell, ufc_simplex  # noqa: F401
from FIAT.hdivcurl import Hdiv, Hcurl                     # noqa: F401

__version__ = pkg_resources.get_distribution("FIAT").version

# List of supported elements and mapping to element classes
supported_elements = {"Argyris": Argyris,
                      "Brezzi-Douglas-Marini": BrezziDouglasMarini,
                      "Brezzi-Douglas-Fortin-Marini": BrezziDouglasFortinMarini,
                      "Bubble": Bubble,
                      "Crouzeix-Raviart": CrouzeixRaviart,
                      "Discontinuous Lagrange": DiscontinuousLagrange,
                      "Discontinuous Taylor": DiscontinuousTaylor,
                      "Discontinuous Raviart-Thomas": DiscontinuousRaviartThomas,
                      "Hermite": CubicHermite,
                      "Lagrange": Lagrange,
                      "Gauss-Lobatto-Legendre": GaussLobattoLegendre,
                      "Gauss-Legendre": GaussLegendre,
                      "Morley": Morley,
                      "Nedelec 1st kind H(curl)": Nedelec,
                      "Nedelec 2nd kind H(curl)": NedelecSecondKind,
                      "Raviart-Thomas": RaviartThomas,
                      "Regge": Regge,
                      "EnrichedElement": EnrichedElement,
                      "NodalEnrichedElement": NodalEnrichedElement,
                      "TensorProductElement": TensorProductElement,
                      "BrokenElement": DiscontinuousElement,
                      "HDiv Trace": HDivTrace,
                      "Hellan-Herrmann-Johnson": HellanHerrmannJohnson}

# List of extra elements
extra_elements = {"P0": P0,
                  "Quintic Argyris": QuinticArgyris}
