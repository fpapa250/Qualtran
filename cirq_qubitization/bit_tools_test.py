import pytest

from cirq_qubitization.bit_tools import (
    iter_bits,
    fixed_point_bits_to_float,
    float_to_fixed_point_bits,
)


def test_iter_bits():
    assert list(iter_bits(0, 2)) == [0, 0]
    assert list(iter_bits(1, 2)) == [0, 1]
    assert list(iter_bits(2, 2)) == [1, 0]
    assert list(iter_bits(3, 2)) == [1, 1]
    with pytest.raises(ValueError):
        assert list(iter_bits(4, 2)) == [1, 0, 0]


@pytest.mark.parametrize(
    "input,expected", [[('1101001', 2), -10.25], [('010101010101', 4), 85.3125]]
)
def test_fixed_point_bits_to_float(input, expected):
    bin_val, frac = input
    val_fixed = fixed_point_bits_to_float(bin_val, frac)
    assert val_fixed == expected



@pytest.mark.parametrize(
    "input,expected", [[(-10.39499592010, 8, 2), '1101001'], [(2**8 / 3.0, 12, 4), '010101010101']]
)
def test_float_to_fixed_point_bits(input, expected):
    val, width, frac = input
    bin_val = float_to_fixed_point_bits(val, width, frac)
    assert bin_val == expected
