"""The content digest keys a kernel on what it is built from, not on the process."""

import numpy as np

from modax._kernel_cache import content_digest


def _make(scale, table):
    def rhs(y, t, p):
        return (-scale * p[0] * y[0] + table[1],)

    def kernel(y):
        return rhs(y, 0.0, y)

    return kernel


def test_the_digest_sees_through_closures():
    table = np.array([1.0, 2.0])
    assert content_digest(_make(2.0, table)) == content_digest(_make(2.0, table.copy()))
    assert content_digest(_make(2.0, table)) != content_digest(_make(3.0, table))
    assert content_digest(_make(2.0, table)) != content_digest(_make(2.0, np.array([1.0, 2.5])))


def test_the_digest_follows_globals_and_attributes():
    def kernel(y):
        return GLOBAL_SCALE * y

    global GLOBAL_SCALE
    GLOBAL_SCALE = 1.0
    one = content_digest(kernel)
    GLOBAL_SCALE = 2.0
    two = content_digest(kernel)
    assert one != two
    kernel._primal = np.zeros(3)
    assert content_digest(kernel) != two
