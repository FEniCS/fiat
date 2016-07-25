How to run regression tests
===========================

To run regression tests with default parameters, simply run::

  cd <fiatdir>/test/regression/
  py.test

Look at test.py for more options.


How to update references
========================

To update the references for the FIAT regression tests, first commit
your changes, then run the regression test (to generate the new
references) and finally run the script upload::

  <commit your changes>
  cd <fiatdir>/test/regression/
  py.test
  ./scripts/upload

Note: You may be asked for your *Bitbucket* username and password when
uploading the reference data, if use of ssh keys fails.

Note: The upload script will push the new references to the
fiat-reference-data repository. This is harmless even if these
references are not needed later.

Note: The upload script will update the file fiat-regression-data-id
and commit this change to the currently active branch, remember to
include this commit when merging or pushing your changes elsewhere.

Note: You can cherry-pick the commit that updated
fiat-regression-data-id into another branch to use the same set of
references there.

Note: If you ever get merge conflicts in the fiat-regression-data-id,
always pick one version of the file. Most likely you'll need to update
the references again.


How to run regression tests against a different set of regression data
======================================================================

To run regression tests and compare to a different set of regression
data, perhaps to see what has changed in generated code since a
certain version, check out the fiat-regression-data-id file you want
and run tests as usual::

  cd <fiatdir>/test/regression/
  git checkout <fiat-commit-id> fiat-regression-data-id
  py.test

The test.py script will run scripts/download which will check out the
regression data with the commit id from fiat-regression-data-id in
fiat-regression-data/. Run::

    DATA_REPO_GIT="" ./scripts/download/

to use https instead of ssh.
