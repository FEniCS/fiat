# Written by Marie E. Rognes (meg@cma.uio.no)
# Dec 17th 2007
# Distributed under the LGPL license

import copy

class Functionaltype(object):
    """
    name         = A name for the type for Functional
    points       = 
    components   =
    weights      =
    derivatives  = 
    mapping      = The appropriate mapping for the functional.
    """
    def __init__(self, name, points=None, directions=None, weights=None,
                 derivatives=None, mapping=None):
        """ Initialize Functionaltype.

        Leave handling of None values to the receiving part."""
        
        if isinstance(name, Functionaltype):
            self.name = name.name
            self.points = copy.deepcopy(name.points)
            self.directions = copy.deepcopy(name.directions)
            self.weights = copy.deepcopy(name.weights)
            self.derivatives = copy.deepcopy(name.derivatives)
            self.mapping = name.mapping
            return
        else:
            self.name = name
            self.points = points
            self.directions = directions
            self.weights = weights
            self.derivatives = derivatives
            self.mapping = mapping
            return

    def __equals__(self, other):
        """ Very simple check whether functionals are of the same
        type. """
        return self.name == other.name

    def get_attributes(self):
        """ Return a list of all attributes """
        return (self.name, self.points, self.directions, self.weights,
                self.derivatives)

    def __str__(self):
        """ First stab at Pretty print """
        label = "%s" % self.name
        if self.directions:
            dirs = " in the directions %s " % [dir for dir in self.directions]
            label += dirs
        if self.points:
            at = " at %s" % str([pt for pt in self.points])
            label += at
        if self.weights:
            w = " [%s] \times " % str([w for w in self.weights])
            w += label
            label = w
        if self.derivatives:
            ds = str(["d/d%s " % str(d) for d in self.derivatives])
            ds += label
            label = ds
        return label


if __name__ == "__main__":
    import shapes 
    import RaviartThomas, Lagrange, BDM, BDFM, CrouzeixRaviart, Nedelec


    elements = [Lagrange.Lagrange(2, 1),
            RaviartThomas.RaviartThomas(2, 1),
            BDM.BDM(2, 1),
            BDFM.BDFM(2, 1),
            CrouzeixRaviart.CrouzeixRaviart(2),
            Nedelec.Nedelec(2, 2),
            ]

    for element in elements:
        print element
        types = element.dual_basis().get_dualbasis_types()
        string = ""
        for type in types:
            string += "%s\n" % type
        print string
        

