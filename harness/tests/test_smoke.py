import unmesh_harness


def test_harness_sees_unmesh():
    assert unmesh_harness.UNMESH_VERSION == "0.1.0"
