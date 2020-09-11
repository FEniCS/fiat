from FIAT.nedelec import Nedelec
from FIAT.reference_element import *

element = DefaultTriangle()

N = Nedelec(element, 2, variant='integral')
