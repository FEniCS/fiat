# A C implementation for ln_gamma function taken from Numerical
# recipes in C: The art of scientific
# computing, 2nd edition, Press, Teukolsky, Vetterling, Flannery, Cambridge
# University press, page 214
# translated into Python by Robert Kirby
# See originally Abramowitz and Stegun's Handbook of Mathematical Functions.

from math import log, exp

def ln_gamma( xx ):
    cof = [76.18009172947146,\
           -86.50532032941677, \
           24.01409824083091, \
           -1.231739572450155, \
           0.1208650973866179e-2, \
           -0.5395239384953e-5 ]
    y = xx
    x = xx
    tmp = x + 5.5
    tmp -= (x + 0.5) * log(tmp)
    ser = 1.000000000190015
    for j in range(0,6):
        y = y + 1
        ser += cof[j] / y
    return -tmp + log( 2.5066282746310005*ser/x )
    
def gamma( xx ):
    return exp( ln_gamma( xx ) )
