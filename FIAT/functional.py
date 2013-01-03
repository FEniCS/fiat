# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
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

# functionals require:
# - a degree of accuracy (-1 indicates that it works for all functions
#   such as point evaluation)
# - a reference element domain
# - type information

import numpy,string

# Import AD modules from ScientificPython
import Scientific.Functions.Derivatives as Derivatives
import Scientific.Functions.FirstDerivatives as FirstDerivatives
from functools import reduce

def index_iterator( shp ):
    """Constructs a generator iterating over all indices in
    shp in generalized column-major order  So if shp = (2,2), then we
    construct the sequence (0,0),(0,1),(1,0),(1,1)"""
    if len(shp) == 0:
        return
    elif len( shp ) == 1:
        for i in range( shp[0] ):
            yield [i]
    else:
        shp_foo = shp[1:]
        for i in range( shp[0] ):
            for foo in index_iterator( shp_foo ):
                yield [i] + foo

# also put in a "jet_dict" that maps
# pt --> {wt, multiindex, comp}
# the multiindex is an iterable of nonnegative
# integers

class Functional:
    """Class implementing an abstract functional.
    All functionals are discrete in the sense that
    the are written as a weighted sum of (components of) their
    argument evaluated at particular points."""
    def __init__( self , ref_el , target_shape , \
                  pt_dict , deriv_dict , functional_type ):
        self.ref_el = ref_el
        self.target_shape = target_shape
        self.pt_dict = pt_dict
        self.deriv_dict = deriv_dict
        self.functional_type = functional_type
        if len(deriv_dict) > 0:
            per_point = reduce( lambda a,b : a + b , list(deriv_dict.values()) )
            alphas = \
                [ foo[1] for foo in per_point ]
            self.max_deriv_order = max( [ sum( foo ) for foo in alphas ] )
        else:
            self.max_deriv_order = 0
        return
    def evaluate( self , f ):
        """Evaluates the functional on some callable object f."""
        result = 0

        # non-derivative part
        for pt in pt_dict:
            wc_list = pt_dict[pt]
            for (w,c) in wc_list:
                if comp == tuple:
                    result += w * f(pt)
                else:
                    result += w * f(pt)[comp]

        for pt in self.deriv_dict:
            dpt = tuple( [ Derivatives.DerivVar( pt[i] , i , self.max_deriv_order ) \
                               for i in range( len( pt ) ) ] )
            for (w,a,c) in self.deriv_dict[pt]:
                fpt = f( dpt )
                order = sum( a )
                if c == tuple():
                    val_cur = fpt[order]
                else:
                    val_cur = fpt[c][order]
                for i in range( len[a] ):
                    for j in range( a[j] ):
                        val_cur = val_cur[i]

                result += val_cur

        return result

    def get_point_dict( self ):
        """Returns the functional information, which is a dictionary
        mapping each point in the support of the functional to a list
        of pairs containing the weight and component."""
        return self.pt_dict
    def get_reference_element( self ):
        """Returns the reference element."""
        return self.ref_el
    def get_type_tag( self ):
        """Returns the type of function (e.g. point evaluation or
        normal component, which is probably handy for clients of FIAT"""
        return self.functional_type

    # overload me in subclasses to make life easier!!
    def to_riesz( self , poly_set ):
        """Constructs an array representation of the functional over
        the base of the given polynomial_set so that f(phi) for any
        phi in poly_set is given by a dot product."""
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        pt_dict = self.get_point_dict()

        pts = list(pt_dict.keys())

        # bfs is matrix that is pdim rows by num_pts cols
        # where pdim is the polynomial dimension

        bfs = es.tabulate( ed , pts )

        result = numpy.zeros( poly_set.coeffs.shape[1:] , "d" )

        shp = poly_set.get_shape()


        # loop over points
        for j in range( len( pts ) ):
            pt_cur = pts[j]
            wc_list = pt_dict[ pt_cur ]

            # loop over expansion functions
            for i in range( bfs.shape[0] ):
                for (w,c) in wc_list:
                    result[c][i] += w * bfs[i,j]

        def pt_to_dpt( pt , dorder ):
            dpt = []
            for i in range( len( pt ) ):
                dpt.append( Derivatives.DerivVar( pt[i] , i , dorder ) )
            return tuple( dpt )

        # loop over deriv points
        dpt_dict = self.deriv_dict
        mdo = self.max_deriv_order

        dpts = list(dpt_dict.keys())
        dpts_dv = [ pt_to_dpt( pt , mdo ) for pt in dpts ]

        dbfs = es.tabulate( ed , dpts_dv )

        for j in range( len( dpts ) ):
            dpt_cur = dpts[j]
            for i in range( dbfs.shape[0] ):
                for (w,a,c) in dpt_dict[ dpt_cur ]:
                    dval_cur = dbfs[i,j][sum(a)]
                    for k in range( len( a ) ):
                        for l in range( a[k] ):
                            dval_cur = dval_cur[k]

                    result[c][i] += w * dval_cur

        return result

    def tostr( self ):
        return self.functional_type


class PointEvaluation( Functional ):
    """Class representing point evaluation of scalar functions at a
    particular point x."""
    def __init__( self , ref_el , x ):
        pt_dict = { x : [ (1.0,tuple()) ] }
        Functional.__init__( self , ref_el , \
                             tuple() , \
                             pt_dict , {} ,
                             "PointEval" )
        return
    def tostr( self ):
        import string
        x = list(map(str,list(self.pt_dict.keys())[0]))
        return "u(%s)"%(string.join(x,","),)

class ComponentPointEvaluation( Functional ):
    """Class representing point evaluation of a particular component
    of a vector function at a particular point x."""
    def __init__( self , ref_el , comp , shp , x ):
        if len( shp ) != 1:
            raise Exception("Illegal shape")
        if comp < 0 or comp >= shp[0]:
            raise Exception("Illegal component")
        self.comp = comp
        pt_dict = { x : [ ( 1.0 , (comp,) ) ] }
        Functional.__init__( self , ref_el , \
                                 shp, pt_dict , {}, \
                                 "ComponentPointEval" )

    def tostr( self ):
        import string
        x = list(map(str,list(self.pt_dict.keys())[0]))
        return "(u[%d](%s)"%(self.comp,string.join(x,","))


class PointDerivative( Functional ):
    """Class representing point partial differentiation of scalar
    functions at a particular point x."""
    def __init__( self , ref_el , x , alpha ):
        dpt_dict = { x : [ (1.0,alpha,tuple()) ] }
        self.alpha = alpha
        self.order = sum( self.alpha )

        Functional.__init__( self , ref_el , tuple() , {} , \
                             dpt_dict , "PointDeriv" )

        return
    def to_riesz( self , poly_set ):
        x = list(self.deriv_dict.keys())[0]
        dx = tuple( [ Derivatives.DerivVar( x[i] , i , self.order ) \
                          for i in range( len( x ) ) ] )

        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()

        bfs = es.tabulate( ed , [ dx ] )[:,0]


        idx = []
        for i in range( len( self.alpha ) ):
            for j in range( self.alpha[i] ):
                idx.append( i )
        idx = tuple( idx )

        return numpy.array( [ numpy.array(b[self.order])[idx] for b in bfs ] )

class PointNormalDerivative( Functional ):
    def __init__( self , ref_el , facet_no , pt ):
        n = ref_el.compute_normal( facet_no )
        self.n = n
        sd = ref_el.get_spatial_dimension()

        alphas = []
        for i in range( sd ):
            alpha = [0]*sd
            alpha[i] = 1
            alphas.append( alpha )

        dpt_dict = { pt : [ (n[i],alphas[i],tuple()) for i in range( sd ) ] }

        Functional.__init__( self , ref_el , tuple() , {} , \
                             dpt_dict , "PointNormalDeriv" )

        return

    def to_riesz( self , poly_set ):
        x = list(self.deriv_dict.keys())[0]
        dx = tuple( [ FirstDerivatives.DerivVar( x[i] , i ) \
                          for i in range( len( x ) ) ] )

        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()

        bfs = es.tabulate( ed , [ dx ] )[:,0]

        bfs_grad = numpy.array( [ b[1] for b in bfs ] )
        return numpy.dot( bfs_grad , self.n )


class IntegralMoment (Functional):
    """
    An IntegralMoment is a functional

    """
    def __init__( self , ref_el , Q , f_at_qpts , comp = tuple() ,
                  shp = tuple()):
        """
        Create IntegralMoment

        *Arguments*

          ref_el
              The reference element (cell)
          Q (QuadratureRule)
              A quadrature rule for the integral
          f_at_qpts
              ???
          comp (tuple)
              A component ??? (Optional)
          shp  (tuple)
              The shape ??? (Optional)
        """
        qpts,qwts = Q.get_points(), Q.get_weights()
        pt_dict = {}
        self.comp = comp
        for i in range( len( qpts ) ):
            pt_cur = tuple(qpts[i])
            pt_dict[ pt_cur ] = [ (qwts[i] * f_at_qpts[i] , comp ) ]
        Functional.__init__( self , ref_el , shp , \
                                 pt_dict , {} , "IntegralMoment" )

    def to_riesz( self , poly_set ):
        T = poly_set.get_reference_element()
        sd = T.get_spatial_dimension()
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        pts = list(self.pt_dict.keys())
        bfs = es.tabulate( ed , pts )
        wts = numpy.array( [ foo[0][0] for foo in list(self.pt_dict.values()) ] )
        result = numpy.zeros( poly_set.coeffs.shape[1:] , "d" )
        result[self.comp,:] = numpy.dot( bfs , wts )
        return result

class FrobeniusIntegralMoment( Functional ):
    def __init__( self , ref_el , Q , f_at_qpts ):
        # f_at_qpts is num components x num_qpts
        if len( Q.get_points() ) != f_at_qpts.shape[1]:
            raise Exception("Mismatch in number of quadrature points and values")

        # make sure that shp is same shape as f given
        shp = (f_at_qpts.shape[0],)

        qpts,qwts = Q.get_points(), Q.get_weights()
        pt_dict = {}
        for i in range( len( qpts ) ):
            pt_cur = tuple(qpts[i])
            pt_dict[pt_cur] = [(qwts[i] * f_at_qpts[j,i] , (j, ) )
                               for j in range(f_at_qpts.shape[0])]

        Functional.__init__( self , ref_el , shp , \
                                 pt_dict , {} , "FrobeniusIntegralMoment" )



# point normals happen on a d-1 dimensional facet
# pt is the "physical" point on that facet
class PointNormalEvaluation( Functional ):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1."""
    def __init__( self , ref_el , facet_no , pt ):
        n = ref_el.compute_normal( facet_no )
        self.n = n
        sd = ref_el.get_spatial_dimension()

        pt_dict = { pt : [ (n[i],(i,)) for i in range( sd ) ] }

        shp = (sd,)
        Functional.__init__( self , ref_el , shp , \
                             pt_dict , {} , "PointNormalEval" )
        return

class PointEdgeTangentEvaluation( Functional ):
    """Implements the evaluation of the tangential component of a
    vector at a point on a facet of dimension 1."""
    def __init__( self , ref_el , edge_no , pt ):
        t = ref_el.compute_edge_tangent( edge_no )
        self.t = t
        sd = ref_el.get_spatial_dimension()
        pt_dict = { pt : [ (t[i],(i,)) for i in range( sd ) ] }
        shp = (sd,)
        Functional.__init__( self , ref_el  ,shp , \
                             pt_dict , {} , "PointEdgeTangent" )
    def tostr( self ):
        import string
        x = list(map(str,list(self.pt_dict.keys())[0]))
        return "(u.t)(%s)"%(string.join(x,","),)
    def to_riesz( self , poly_set ):
        # should be singleton
        xs = list(self.pt_dict.keys())
        phis = poly_set.get_expansion_set().tabulate( poly_set.get_embedded_degree() , xs )
        return numpy.outer( self.t , phis )


class PointFaceTangentEvaluation( Functional ):
    """Implements the evaluation of a tangential component of a
    vector at a point on a facet of codimension 1."""
    def __init__( self , ref_el , face_no , tno , pt ):
        t = ref_el.compute_face_tangents( face_no )[ tno ]
        self.t = t
        self.tno = tno
        sd = ref_el.get_spatial_dimension()
        pt_dict = { pt : [ (t[i],(i,)) for i in range( sd ) ] }
        shp = (sd,)
        Functional.__init__( self , ref_el  ,shp , \
                             pt_dict , {} , "PointFaceTangent" )
    def tostr( self ):
        import string
        x = list(map(str,list(self.pt_dict.keys())[0]))
        return "(u.t%d)(%s)"%(self.tno,string.join(x,","),)
    def to_riesz( self , poly_set ):
        xs = list(self.pt_dict.keys())
        phis = poly_set.get_expansion_set().tabulate( poly_set.get_embedded_degree() , xs )
        return numpy.outer( self.t , phis )

class PointScaledNormalEvaluation( Functional ):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1, where the normal is scaled by
    the volume of that facet."""
    def __init__( self , ref_el , facet_no , pt ):
        self.n = ref_el.compute_scaled_normal( facet_no )
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        pt_dict = { pt : [ (self.n[i],(i,)) for i in range( sd ) ] }
        Functional.__init__( self , ref_el , shp , \
                             pt_dict , {} , "PointScaledNormalEval" )
        return
    def tostr( self ):
        import string
        x = list(map(str,list(self.pt_dict.keys())[0]))
        return "(u.n)(%s)"%(string.join(x,","),)

    def to_riesz( self , poly_set ):
        xs = list(self.pt_dict.keys())
        phis = poly_set.get_expansion_set().tabulate( poly_set.get_embedded_degree() , xs )

        return numpy.outer( self.n , phis )



def moments_against_set( ref_el , U , Q ):
    # check that U and Q are both over ref_el

    qpts = Q.get_points()
    qwts = Q.get_weights()

    Uvals = U.tabulate( pts )

    # handle scalar case

    for i in range( Uvals.shape[0] ):  # loop over members of U
        pass



if __name__=="__main__":
    # test functionals
    from . import polynomial_set, reference_element
    ref_el = reference_element.DefaultTriangle()
    sd = ref_el.get_spatial_dimension()
    U = polynomial_set.ONPolynomialSet( ref_el , 5 )

    f = PointDerivative( ref_el , (0.0,0.0) , (1,0) )
    print(numpy.allclose( Functional.to_riesz( f , U ) , f.to_riesz( U ) ))

    f = PointNormalDerivative( ref_el , 0 , (0.0,0.0) )
    print(numpy.allclose( Functional.to_riesz( f , U ) , f.to_riesz( U ) ))
