"""factorial.py.  It does what you think."""

def factorial( n ):
    """Computes n! for n an integer >= 0.
    Raises an ArithmeticError otherwise."""
    if type(n) != type(1) or n < 0:
        raise ArithmeticError, "factorial only defined on natural numbers."
    f = 1
    for i in xrange(1,n+1):
        f = f * i
    return f

