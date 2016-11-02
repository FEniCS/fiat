===========================
Changes in the next release
===========================


Summary of changes
==================

.. note:: Developers should use this page to track and list changes
          during development. At the time of release, this page should
          be published (and renamed) to list the most important
          changes in the new release.

- More elegant edge-based degrees of freedom are used for generalized Regge
  finite elements.  This is a internal change and is not visible to other parts
  of FEniCS.
- The name of the mapping for generalized Regge finite element is changed to
  "double covariant piola" from "pullback as metric". Geometrically, this
  mapping is just the pullback of covariant 2-tensor fields in terms of proxy
  matrix-fields. Because the mapping for 1-forms in FEniCS is currently named
  "covariant piola", this mapping for symmetric tensor product of 1-forms is
  thus called "double covariant piola". This change causes multiple internal
  changes downstream in UFL and FFC. But this change should not be visible to
  the end-user.
- Added support for the Hellan-Herrmann-Johnson element (symmetric matrix
  fields with normal-normal continuity in 2D).
- Add method ``FiniteElement.is_nodal()`` for checking element nodality
- Add ``NodalEnrichedElement`` which merges dual bases (nodes) of given
  elements and orthogonalizes basis for nodality
- Restructuring ``finite_element.py`` with the addition of a non-nodal class
  ``FiniteElement`` and a nodal class ``CiarletElement``. ``FiniteElement`` is
  designed to be used to create elements where, in general, a nodal basis isn't
  well-defined. ``CiarletElement`` implements the usual nodal abstraction of
  a finite element.
- Removing ``trace.py`` and ``trace_hdiv.py`` with a new implementation of the
  trace element of an HDiv-conforming element: ``HDivTrace``. It is also
  mathematically equivalent to the former ``DiscontinuousLagrangeTrace``, that
  is, the DG field defined only on co-dimension 1 entities.
- All nodal finite elements inherit from ``CiarletElement``, and the non-nodal
  ``TensorProductElement``, ``EnrichedElement`` and ``HDivTrace`` inherit from
  ``FiniteElement``.

Detailed changes
================

.. note:: At the time of release, make a verbatim copy of the
          ChangeLog here (and remove this note).
