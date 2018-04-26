Changelog
=========

2018.1.0 (2018-05-01)
-------------

- Remove python2 support.

2017.2.0 (2017-12-05)
---------------------

- Add quadrilateral and hexahedron reference cells
- Add quadrilateral and hexahedron elements (with a wrapping class for TensorProductElement)

2017.1.0.post1 (2017-09-12)
---------------------------

- Change PyPI package name to fenics-fiat.

2017.1.0 (2017-05-09)
---------------------

- Extended the discontinuous trace element ``HDivTrace`` to support tensor
  product reference cells. Tabulating the trace defined on a tensor product
  cell relies on the argument ``entity`` to specify a facet of the cell. The
  backwards compatibility case ``entity=None`` does not support tensor product
  tabulation as a result. Tabulating the trace of triangles or tetrahedron
  remains unaffected and works as usual with or without an entity argument.

2016.2.0 (2016-11-30)
---------------------

- Enable Travis CI on GitHub
- Add Firedrake quadrilateral cell
- Add tensor product cell
- Add facet -> cell coordinate transformation
- Add Bubble element
- Add discontinuous Taylor element
- Add broken element and H(div) trace element
- Add element restrictions onto mesh entities
- Add tensor product elements (for tensor product cells)
- Add H(div) and H(curl) element-modifiers for TPEs
- Add enriched element, i.e. sum of elements (e.g. for building Mini)
- Add multidimensional taylor elements
- Add Gauss Lobatto Legendre elements
- Finding non-vanishing DoFs on a facets
- Add tensor product quadrature rule
- Make regression tests working again after few years
- Prune modules having only ``__main__`` code including
  transform_morley, transform_hermite
  (ff86250820e2b18f7a0df471c97afa87207e9a7d)
- Remove newdubiner module (b3b120d40748961fdd0727a4e6c62450198d9647,
  reference removed by cb65a84ac639977b7be04962cc1351481ca66124)
- Switch from homebrew factorial/gamma to math module (wraps C std lib)

2016.1.0 (2016-06-23)
---------------------

- Minor fixes

1.6.0 (2015-07-28)
------------------

- Support DG on facets through the element "Discontinuous Lagrange
  Trace"

1.5.0 (2015-01-12)
------------------

- Require Python 2.7
- Python 3 support
- Remove ScientificPython dependency and add dependency on SymPy

1.4.0 (2014-06-02)
------------------

- Support discontinuous/broken Raviart-Thomas

1.3.0 (2014-01-07)
------------------

- Version bump.

1.1.0 (2013-01-07)
------------------

- Support second kind Nedelecs on tetrahedra over degree >= 2
- Support Brezzi-Douglas-Fortin-Marini elements (of degree 1, 2), again

1.0.0 (2011-12-07)
------------------

- No changes since 1.0-beta, only updating the version number

1.0-beta (2011-08-11)
---------------------

- Change of license to LGPL v3+
- Minor fixes

0.9.9 (2011-02-23)
------------------

- Add ``__version__``
- Add second kind Nedeles on triangles

0.9.2 (2010-07-01)
------------------

- Bug fix for 1D quadrature

0.9.1 (2010-02-03)
------------------

- Cleanups and small fixes

0.9.0 (2010-02-01)
------------------

- New improved interface with support for arbitrary reference elements

0.3.5
-----

0.3.4
-----

0.3.3
-----

- Bug fix in Nedelec
- Support for ufc element

0.3.1
-----

- Bug fix in DOF orderings for H(div) elements
- Preliminary type system for DOF
- Allow user to change ordering of reference dof
- Brezzi-Douglas-Fortin-Marini elements working

0.3.0
-----

- Small changes to H(div) elements preparing for integration with FFC
- Switch to numpy
- Added primitive testing harness in fiat/testing

0.2.4
-----

- Fixed but in P0.py

0.2.3
-----

- Updated topology/ geometry so to allow different orderings of entities

0.2.2
-----

- Added Raviart-Thomas element, verified RT0 against old version of code
- Started work on BDFM, Nedelec (not working)
- Fixed projection, union of sets (error in SVD usage)
- Vector-valued spaces have general number of components
