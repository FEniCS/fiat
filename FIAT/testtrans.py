import shapes,transformedspace,Lagrange,numpy,RaviartThomas,BDFM

def test_scalar():
    pts = shapes.make_lattice( 2 , 1 )
    newverts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))
    U = Lagrange.Lagrange(2,1).function_space()
    Utrans = transformedspace.AffineTransformedFunctionSpace( U , newverts )
    Ujet = U.tabulate_jet( 1 , pts )
    Utransjet = Utrans.tabulate_jet( 2 , newverts )
    edge_pts_transformed = ((1./3.,),(2./3.,))
    edge_pts = ((-1./3,),(1./3.,))
    Utracejet = U.trace_tabulate_jet( 1 , 1 , 1 , edge_pts )
    for i in range(len( Utracejet )):
        for k in Utracejet[i]:
            print k
            print Utracejet[i][k]
    trans_ref_verts = ((0.0,),(1.0,))
    Utranstracejet = Utrans.trace_tabulate_jet( 1 , 1 , 1 , \
                                                edge_pts_transformed , \
                                                trans_ref_verts )
    for i in range(len( Utranstracejet )):
        for k in Utranstracejet[i]:
            print k
            print Utranstracejet[i][k]


    print "deriv mats"
    for i in range(2):
        print U.base.dmats[i]
        print Utrans.dmats[i]
        print
    return

def test_piola( ):
    pts = shapes.make_lattice( 2 , 1 )
    newverts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))
    U=RaviartThomas.RaviartThomas(2,0).function_space()
    V=BDFM.BDFM(2,1).function_space()
    Utrans = transformedspace.PiolaTransformedFunctionSpace(U,newverts,"div")
    Vtrans = transformedspace.PiolaTransformedFunctionSpace(V,newverts,"div")

    newpts = ((0.5,0.5),(0.0,0.5),(0.5,0.0))

    edge_pts_transformed = ((1./3.,),(2./3.,))
    edge_pts = ((-1./3,),(1./3.,))

    Ujet = Utrans.tabulate_jet( 1 , newpts )
    Vjet = Vtrans.tabulate_jet( 1 , newpts )

    Utrjet = Utrans.trace_tabulate_jet(1,1,1, edge_pts_transformed, \
                                       ((0.0,),(1.0,)))
    Vtrjet = Vtrans.trace_tabulate_jet(1,1,1, edge_pts_transformed, \
                                       ((0.0,),(1.0,)))

    U0vals = Ujet[0][0][(0,0)]
    U1vals = Ujet[1][0][(0,0)]

    V0vals = Vjet[0][0][(0,0)]
    V1vals = Vjet[1][0][(0,0)]

##    print "X and Y components of vectors agree?"
##    print numpy.allclose( U0vals , V0vals )
##    print numpy.allclose( U1vals , V1vals )
##
##    print "X-partials of X and Y components agree?"
##    print numpy.allclose( Ujet[0][1][(1,0)] , Vjet[0][1][(1,0)] )
##    print numpy.allclose( Ujet[0][1][(0,1)] , Vjet[0][1][(0,1)] )
##
##    print "Y-partials of X and Y components agree?"
##    print numpy.allclose( Ujet[1][1][(1,0)] , Vjet[1][1][(1,0)] )
##    print numpy.allclose( Ujet[1][1][(0,1)] , Vjet[1][1][(0,1)] )
##
##    print "X and Y components of trace agree?"
##    print numpy.allclose( Utrjet[0][0][(0,0)] , Vtrjet[0][0][(0,0)] )
##    print numpy.allclose( Utrjet[1][0][(0,0)] , Vtrjet[1][0][(0,0)] )
##
##    print "X-partials of trace X and Y components agree?"
##    print numpy.allclose( Utrjet[0][1][(1,0)] , Vtrjet[0][1][(1,0)] )
##    print numpy.allclose( Utrjet[0][1][(0,1)] , Vtrjet[0][1][(0,1)] )
##
##    print "Y-partials of trace X and Y components agree?"
##    print numpy.allclose( Utrjet[1][1][(1,0)] , Vtrjet[1][1][(1,0)] )
##    print numpy.allclose( Utrjet[1][1][(0,1)] , Vtrjet[1][1][(0,1)] )

#    print len( U.base.dmats )
#    print len( Utrans.dmats )
#    print Utrans.spatial_dimension()

    for dm in U.base.dmats:
        print dm
        print
    print
    print
    for dm in Utrans.dmats:
        print dm
        print

def test_poly_scalar():
    U = Lagrange.Lagrange(2,1).function_space()
    pts = shapes.make_lattice( 2 , 1 )
    newverts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))
    Utr = transformedspace.AffineTransformedFunctionSpace( U , newverts )
    print [ Utr[0](v) for v in newverts ]
    print [ U[0](v) for v in pts ]
    print [ [ Utr[0].deriv(i)(v) for v in newverts ] for i in (0,1)]
    print [ [ U[0].deriv(i)(v) for v in newverts ] for i in (0,1) ]
    

def test_poly_piola( ):
    U = RaviartThomas.RaviartThomas(2,1).function_space()
    newverts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))
    Utr = transformedspace.PiolaTransformedFunctionSpace( U , newverts , "div" )
    print Utr.eval_all( newverts[0] )
    print Utr[0](newverts[0])

def testdmats( ):
    U = Lagrange.Lagrange(2,1).function_space()
    newverts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))
    Utr = transformedspace.AffineTransformedSpace( U , newverts )
    print Utr.dmats
    print U.base.dmats

if __name__=="__main__":
    #test_scalar()
    test_piola()
    #test_poly_scalar()
    #test_poly_piola()
