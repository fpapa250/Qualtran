from typing import Iterator
import math
from unicodedata import digit


def iter_bits(val: int, width: int) -> Iterator[int]:
    """Iterate over the bits in a binary representation of `val`.

    This uses a big-endian convention where the most significant bit
    is yielded first.

    Args:
        val: The integer value. Its bitsize must fit within `width`
        width: The number of output bits.
    """
    if val.bit_length() > width:
        raise ValueError(f"{val} exceeds width {width}.")
    for b in f'{val:0{width}b}':
        yield int(b)


def float_to_fixed_point_bits(val: float, width: int, num_fractional_bits: int) -> str:
    scale_factor = 2**num_fractional_bits
    scaled_val = math.floor(abs(val * scale_factor))
    fixed_point = bin(scaled_val)[2:2+width-1]
    if len(fixed_point) < width:
        fixed_point = '0'*(width-len(fixed_point)) + fixed_point
    if val < 0:
        return '1' + fixed_point
    else:
        return '0' + fixed_point

def fixed_point_bits_to_float(bits: str, num_fractional_bits) -> float:
    sign = (-1)**(int(bits[0]))
    number = 0.0
    integer_width = len(bits) - num_fractional_bits - 1
    for indx, bit in enumerate(bits[1:]):
        power = (integer_width - indx-1)
        if int(bit):
            number += 2**power
    return sign * number