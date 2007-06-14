"""Numbering schemes for entities on edges and faces. The "default"
numbering scheme gives the numbering scheme used in FIAT 0.3.0 and
the "UFC" numbering scheme gives the numbering scheme defined in the
UFC manual (this is used by FFC and DOLFIN)."""

numbering_scheme = "default"

def get_entities():
    global numbering_scheme

    if numbering_scheme == "default":
        
        triangle_edges = { 0 : ( 1 , 2 ) , \
                           1 : ( 2 , 0 ) , \
                           2 : ( 0 , 1 ) }
        
        tetrahedron_edges = { 0 : ( 1 , 2 ) , \
                              1 : ( 2 , 0 ) , \
                              2 : ( 0 , 1 ) , \
                              3 : ( 0 , 3 ) , \
                              4 : ( 1 , 3 ) , \
                              5 : ( 2 , 3 ) }
        
        tetrahedron_faces = { 0 : ( 1 , 3 , 2 ) , \
                              1 : ( 2 , 3 , 0 ) , \
                              2 : ( 3 , 1 , 0 ) , \
                              3 : ( 0 , 1 , 2 ) }
        
    elif numbering_scheme == "UFC":
        
        triangle_edges = { 0 : ( 1 , 2 ) , \
                           1 : ( 0 , 2 ) , \
                           2 : ( 0 , 1 ) }
        
        tetrahedron_edges = { 0 : ( 2 , 3 ) , \
                              1 : ( 1 , 3 ) , \
                              2 : ( 1 , 2 ) , \
                              3 : ( 0 , 3 ) , \
                              4 : ( 0 , 2 ) , \
                              5 : ( 0 , 1 ) }
        
        tetrahedron_faces = { 0 : ( 1 , 2 , 3 ) , \
                              1 : ( 0 , 2 , 3 ) , \
                              2 : ( 0 , 1 , 3 ) , \
                              3 : ( 0 , 1 , 2 ) }
                        
    else:
        raise RuntimeError, "Unknown numbering scheme: " + str(scheme)

    return (triangle_edges, tetrahedron_edges, tetrahedron_faces)
