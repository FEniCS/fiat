"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

__version__ = "1.3.0"

# Version number
FIAT_VERSION = __version__

# Import finite element classes
from FIAT.finite_element import FiniteElement
from FIAT.argyris import Argyris
from FIAT.argyris import QuinticArgyris
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.brezzi_douglas_fortin_marini import BrezziDouglasFortinMarini
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.hermite import CubicHermite
from FIAT.lagrange import Lagrange
from FIAT.morley import Morley
from FIAT.nedelec import Nedelec
from FIAT.nedelec_second_kind import NedelecSecondKind
from FIAT.P0 import P0
from FIAT.raviart_thomas import RaviartThomas
from FIAT.crouzeix_raviart import CrouzeixRaviart

# List of supported elements and mapping to element classes
supported_elements = {"Argyris":                  Argyris,
                      "Brezzi-Douglas-Marini":    BrezziDouglasMarini,
                      "Brezzi-Douglas-Fortin-Marini":    BrezziDouglasFortinMarini,
                      "Crouzeix-Raviart":         CrouzeixRaviart,
                      "Discontinuous Lagrange":   DiscontinuousLagrange,
                      "Hermite":                  CubicHermite,
                      "Lagrange":                 Lagrange,
                      "Morley":                   Morley,
                      "Nedelec 1st kind H(curl)": Nedelec,
                      "Nedelec 2nd kind H(curl)": NedelecSecondKind,
                      "Raviart-Thomas":           RaviartThomas}

# List of extra elements
extra_elements = {"P0":              P0,
                  "Quintic Argyris": QuinticArgyris}

# Important functionality
from .quadrature import make_quadrature
from .reference_element import ufc_simplex
