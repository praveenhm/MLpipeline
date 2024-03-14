"""
Metadata tests.
"""

import os

from metadata import Metadata

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_metadata():
    m = Metadata(os.path.join(dir_path, "test/test.metadata.json"))
    assert m


def test_metadata_allow_page():
    m = Metadata(os.path.join(dir_path, "test/test.metadata.json"))
    assert m

    assert len(m.meta.read) == 2

    assert m.allow_page(10)
    assert not m.allow_page(35)


def test_metadata_allow_page2():
    m = Metadata(os.path.join(dir_path, "test/hr.metadata.json"))
    assert m

    print(m.meta.read)
    assert len(m.meta.read) == 1

    assert m.allow_page(10)
    assert not m.allow_page(8)
