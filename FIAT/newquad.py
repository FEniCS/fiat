import xpermutations, Numeric, quadrature

# classes for Cools' symmetric rules
# need way of handling permutation and rotation symmetry
# The problem is duplication of data
# I need functions for each kind of symmetry that take
# the (x,y) coordinate point and return the list of all points
# generated

eps = 1.e-8

# converts from Cartesian to barycentric coordinates in
# arbitrary spatial dimensions, then if the final coordinate is
# close enough to any of the other coordinates, it sets them to
# be equal.  This avoids roundoff error in the case where exactness
# is important (such as when rotation or permutation should generate
# a point with multiplicity)
def cart_to_lam( pt ):
	lam = list( pt )
	lam.append( 1.0 - sum( lam ) )
	for i in range( len( pt ) ):
		if abs( lam[-1] - lam[i] ) < eps:
			lam[-1] = lam[i]
			break
	return lam

# converts from barycentric to Cartesian coordinates
def lam_to_cart( lam ):
	return lam[:-1]

def fully_symmetric( pt ):
	lam = cart_to_lam( pt )
	unique = {}
	for l in xpermutations.xpermutations( lam ):
		tup = tuple( l )
		if tup not in unique:
			unique[tup] = None
	return map( lam_to_cart , unique.keys() )	

def RO3( pt ):
	lam = cart_to_lam( pt )
	unique = {}
	lam_cur = lam
	for i in range(len(lam)):
		tup = tuple( lam_cur )
		if tup not in unique:
			unique[tup] = None
		lam_cur = lam_cur[1:] + lam_cur[0:1]

	return map(lam_to_cart,unique.keys())

generate = { "fs" : fully_symmetric , "ro3" : RO3 }

# Now, a rule is specified by having a list of tuples of the form
# (w,sym_type,pt) where
# w is the quadrature weight
# sym_type is the string "fs" or "ro3" for fully symmetric
# or ro3-invariant
# pt is the generator



cools_tri_1_1 = [(0.5,"fs",( 0.33333333333333333, 0.33333333333333333) ) ]
cools_tri_2_3 = [(0.16666666666666666,"fs",(0.5,0.5)) ]
cools_tri_4_6 = [(0.054975871827660933,"fs",(0.091576213509770743,0.091576213509770743)), \
				 (0.11169079483900573,"fs",(0.44594849091596488,0.44594849091596488))]
cools_tri_5_7 = [(0.1125,"fs",(0.33333333333333333,0.33333333333333333)),\
				 (0.062969590272413576,"fs",(0.10128650732345633,0.10128650732345633)),\
				 (0.066197076394253090,"fs",(0.47014206410511508,0.47014206410511508))]
cools_tri_6_12 = [(0.025422453185103408,"fs",(0.063089014491502228, 0.063089014491502228)),\
				  (0.058393137863189683,"fs",(0.24928674517091042, 0.24928674517091042)),\
				  (0.041425537809186787,"fs",(0.053145049844816947, 0.31035245103378440))]
cools_tri_7_12 = [(0.026517028157436251,"ro3",(0.062382265094402118, 0.067517867073916085)),\
				  (0.043881408714446055,"ro3",(0.055225456656926611, 0.32150249385198182)),\
				  (0.028775042784981585,"ro3",(0.034324302945097146, 0.66094919618673565)),\
				  (0.067493187009802774,"ro3",(0.51584233435359177, 0.27771616697639178))]
cools_tri_8_16 = [(0.072157803838893584,"fs",(0.33333333333333333, 0.33333333333333333)),\
				  (0.051608685267359125,"fs",(0.17056930775176020, 0.17056930775176020)),\
				  (0.016229248811599040,"fs",(0.050547228317030975, 0.050547228317030975)),\
				  (0.047545817133642312,"fs",(0.45929258829272315, 0.45929258829272315)),\
				  (0.013615157087217497,"fs",(0.72849239295540428, 0.26311282963463811))]


def make_rule( l_of_t ):
	wts = []
	pts = []
	for t in l_of_t:
		w,g,pt = t
		new_pts = generate[g]( pt )
		wts.extend( [ w for foo in new_pts ] )
		pts.extend( new_pts )

# Cools lists everything on the [0,1] element, and we
# need to convert to [-1,1] to be heh heh cool

	wts_big = 2.0 * Numeric.array( wts )
	pts_big = [ tuple( [ 2.0*x-1 for x in pt ] ) for pt in pts ]

	return quadrature.QuadratureRule( pts_big , wts_big )

def main():
	for foo in [cools_tri_1_1,cools_tri_2_3,cools_tri_4_6,\
	            cools_tri_5_7,cools_tri_6_12,cools_tri_7_12,\
	            cools_tri_8_16]:
		Q = make_rule( foo )
		print len(Q.x),len(Q.w)
		print Q.x
		print Q.w

if __name__ == "__main__":
	main()