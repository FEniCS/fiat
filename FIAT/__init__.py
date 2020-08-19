"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

import pkg_resources

# Import finite element classes
from FIAT.finite_element import FiniteElement, CiarletElement  # noqa: F401
from FIAT.argyris import Argyris
from FIAT.bernstein import Bernstein
from FIAT.bell import Bell
from FIAT.argyris import QuinticArgyris
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.brezzi_douglas_fortin_marini import BrezziDouglasFortinMarini
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.discontinuous_taylor import DiscontinuousTaylor
from FIAT.discontinuous_raviart_thomas import DiscontinuousRaviartThomas
from FIAT.serendipity import Serendipity
from FIAT.discontinuous_pc import DPC
from FIAT.hermite import CubicHermite
from FIAT.lagrange import Lagrange
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
from FIAT.gauss_legendre import GaussLegendre
from FIAT.gauss_radau import GaussRadau
from FIAT.morley import Morley
from FIAT.nedelec import Nedelec
from FIAT.nedelec_second_kind import NedelecSecondKind
from FIAT.P0 import P0
from FIAT.raviart_thomas import RaviartThomas
from FIAT.crouzeix_raviart import CrouzeixRaviart
from FIAT.regge import Regge
from FIAT.hellan_herrmann_johnson import HellanHerrmannJohnson
from FIAT.arnold_winther import ArnoldWinther
from FIAT.arnold_winther import ArnoldWintherNC
from FIAT.mardal_tai_winther import MardalTaiWinther
from FIAT.bubble import Bubble, FacetBubble
from FIAT.tensor_product import TensorProductElement
from FIAT.enriched import EnrichedElement
from FIAT.nodal_enriched import NodalEnrichedElement
from FIAT.discontinuous import DiscontinuousElement
from FIAT.hdiv_trace import HDivTrace
from FIAT.mixed import MixedElement                       # noqa: F401
from FIAT.restricted import RestrictedElement             # noqa: F401
from FIAT.quadrature_element import QuadratureElement     # noqa: F401

# Important functionality
from FIAT.quadrature import make_quadrature               # noqa: F401
from FIAT.quadrature_schemes import create_quadrature     # noqa: F401
from FIAT.reference_element import ufc_cell, ufc_simplex  # noqa: F401
from FIAT.hdivcurl import Hdiv, Hcurl                     # noqa: F401

__version__ = pkg_resources.get_distribution("fenics-fiat").version

# List of supported elements and mapping to element classes
supported_elements = {"Argyris": Argyris,
                      "Bell": Bell,
                      "Bernstein": Bernstein,
                      "Brezzi-Douglas-Marini": BrezziDouglasMarini,
                      "Brezzi-Douglas-Fortin-Marini": BrezziDouglasFortinMarini,
                      "Bubble": Bubble,
                      "FacetBubble": FacetBubble,
                      "Crouzeix-Raviart": CrouzeixRaviart,
                      "Discontinuous Lagrange": DiscontinuousLagrange,
                      "S": Serendipity,
                      "DPC": DPC,
                      "Discontinuous Taylor": DiscontinuousTaylor,
                      "Discontinuous Raviart-Thomas": DiscontinuousRaviartThomas,
                      "Hermite": CubicHermite,
                      "Lagrange": Lagrange,
                      "Gauss-Lobatto-Legendre": GaussLobattoLegendre,
                      "Gauss-Legendre": GaussLegendre,
                      "Gauss-Radau": GaussRadau,
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
                      "Hellan-Herrmann-Johnson": HellanHerrmannJohnson,
                      "Conforming Arnold-Winther": ArnoldWinther,
                      "Nonconforming Arnold-Winther": ArnoldWintherNC,
                      "Mardal-Tai-Winther": MardalTaiWinther}

# List of extra elements
extra_elements = {"P0": P0,
                  "Quintic Argyris": QuinticArgyris}
