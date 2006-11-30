import FIAT.shapes, FIAT.Lagrange, FIAT.P0
import cPickle,numpy

def build_matrices():
    L = FIAT.Lagrange.Lagrange
    mats = {}
    for shape in (1,2,3):
        mats[shape] = {}
        U = FIAT.P0.P0(shape).function_space()
        pts = FIAT.shapes.make_lattice(shape,1)
        mats[shape][0] = U.tabulate(pts)
        for degree in range(1,4):
            U = L(shape,degree).function_space()
            pts = FIAT.shapes.make_lattice(shape,degree)
            mats[shape][degree] = U.tabulate(pts)

    return mats

def write_matrices( ):
    mats = build_matrices()
    f = open( "verify.dat" , "w" )
    p = cPickle.Pickler(f)
    p.dump(mats)
    f.close()

def read_matrices( ):
    f = open( "verify.dat" )
    p = cPickle.Unpickler(f)
    return p.load()

def compare_recursive_dictionary_of_mats( A , B ):
    if type(A) != type(B):
        raise RuntimeError, "can't compare objects"
    if type(A) == type({}):
        for i,a in A.iteritems():
            if not compare_recursive_dictionary_of_mats(a,B[i]):
                return False
    else:
        return numpy.allclose(A,B)
    return True
    

def validate( ):
    good_mats = read_matrices()
    new_mats = build_matrices()
    return compare_recursive_dictionary_of_mats(read_matrices(),\
                                                build_matrices())
    

if __name__ == "__main__":
    print validate()
    
