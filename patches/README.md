# Optional scalar-CV ordering correction

The patch targets the root src/moham.cc (SHA-256 42d6e90175b07e957024711b8734c0b4b277bc5a3f4698b81dff416a9fdd701a). It replaces the pairwise epsilon comparison by exact finite-CV ordering, then retains the profile/fingerprint tie-breaks. Default source and paper results are unchanged.

Check applicability with `git apply --check patches/scalar_cv_strict_order.patch`. Apply in a separate version with `git apply patches/scalar_cv_strict_order.patch`.

Compile the isolated check with `g++ -std=c++17 -O2 tests/scalar_cv_order_regression.cpp -o /tmp/camo-cv-test`, then run it. It tests the comparator relation; it does not rerun optimization.
