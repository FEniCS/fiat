"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

# Import finite element classes
from FIAT_NEW.argyris import Argyris
from FIAT_NEW.argyris import QuinticArgyris
from FIAT_NEW.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT_NEW.discontinuous_lagrange import DiscontinuousLagrange
from FIAT_NEW.hermite import CubicHermite
from FIAT_NEW.lagrange import Lagrange
from FIAT_NEW.morley import Morley
from FIAT_NEW.nedelec import Nedelec
from FIAT_NEW.P0 import P0
from FIAT_NEW.raviart_thomas import RaviartThomas

# List of supported elements and mapping to element classes
element_classes = {"Argyris":                  Argyris,
                   "Quintic Argyris":          QuinticArgyris,
                   "Brezzi-Douglas-Marini":    BrezziDouglasMarini,
                   "Discontinuous Lagrange":   DiscontinuousLagrange,
                   "Cubic Hermite":            CubicHermite,
                   "Lagrange":                 Lagrange,
                   "Morley":                   Morley,
                   "Nedelec 1st kind H(curl)": Nedelec,
                   "P0":                       P0,
                   "Raviart-Thomas":           RaviartThomas}

# Important functionality
from quadrature import make_quadrature
