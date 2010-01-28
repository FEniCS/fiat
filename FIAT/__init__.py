"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

# Import finite element classes
from FIAT.argyris import Argyris
from FIAT.argyris import QuinticArgyris
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.hermite import CubicHermite
from FIAT.lagrange import Lagrange
from FIAT.morley import Morley
from FIAT.nedelec import Nedelec
from FIAT.P0 import P0
from FIAT.raviart_thomas import RaviartThomas
from FIAT.crouzeix_raviart import CrouzeixRaviart

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
                   "Raviart-Thomas":           RaviartThomas,
                   "Crouzeix-Raviart":         CrouzeixRaviart}

# Important functionality
from quadrature import make_quadrature
