import shapes,transformedspace,Lagrange,numpy

pts = shapes.make_lattice( 2 , 1 )

newverts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))

U = Lagrange.Lagrange(2,1).function_space()

Utrans = transformedspace.AffineTransformedFunctionSpace( U , newverts )

Ujet = U.tabulate_jet( 1 , pts )

Utransjet = Utrans.tabulate_jet( 2 , newverts )


## for i in range(len( Ujet )):
##     for k in Ujet[i]:
##         print k
##         print Ujet[i][k]


## for i in range(len( Utransjet )):
##     for k in Utransjet[i]:
##         print k
##         print Utransjet[i][k]


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

