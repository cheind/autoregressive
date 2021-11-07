from ..wave import compute_receptive_field


def test_receptive_field():
    assert compute_receptive_field(dilation_seq=[1]) == 2
    assert compute_receptive_field(dilation_seq=[1, 2]) == 4
    assert compute_receptive_field(dilation_seq=[1, 2, 4]) == 8
    assert compute_receptive_field(dilation_seq=[1, 2, 4, 1]) == 9
    assert compute_receptive_field(dilation_seq=[1, 2, 4, 1, 2, 4]) == 15
