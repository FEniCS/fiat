===========================
Changes in the next release
===========================


Summary of changes
==================

.. note:: Developers should use this page to track and list changes
          during development. At the time of release, this page should
          be published (and renamed) to list the most important
          changes in the new release.

- Extended the discontinuous trace element ``HDivTrace`` to support tensor
  product reference cells. Tabulating the trace defined on a tensor product
  cell relies on the argument ``entity`` to specify a facet of the cell. The
  backwards compatibility case ``entity=None`` does not support tensor product
  tabulation as a result. Tabulating the trace of triangles or tetrahedron
  remains unaffected and works as usual with or without an entity argument.

Detailed changes
================

.. note:: At the time of release, make a verbatim copy of the
          ChangeLog here (and remove this note).
