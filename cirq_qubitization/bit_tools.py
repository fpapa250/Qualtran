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
    """Gives binary representation of a float using fixed point precision.

    This uses a signed bit representation rather than 2's complement i.e.

    Args:
        val: The floating point.
        width: The total number of bits uses to represent the float (sign bit, integer bits, fractional bits).
        num_fractional_bits: The number of bits to represent the fractional part of the number

    Returns:
        fixed_point_str: The binary representation of the fixed point value.
    """
    scale_factor = 2**num_fractional_bits
    scaled_val = math.floor(abs(val * scale_factor))
    fixed_point_str = bin(scaled_val)[2:2+width-1]
    if len(fixed_point_str) < width:
        fixed_point = '0'*(width-len(fixed_point_str)) + fixed_point_str
    if val < 0:
        return '1' + fixed_point_str
    else:
        return '0' + fixed_point_str

def fixed_point_bits_to_float(bits: str, num_fractional_bits) -> float:
    """Gives fixed point representation of float given its bitstring representation.

    This uses a signed bit representation rather than 2's complement i.e.

    Args:
        bits: The bitstring representing the floating point value.
        num_fractional_bits: The number of bits to represent the fractional part of the number

    Returns:
        fixed_point_val: The binary representation of the fixed point value.
    """
    sign = (-1)**(int(bits[0]))
    number = 0.0
    integer_width = len(bits) - num_fractional_bits - 1
    for indx, bit in enumerate(bits[1:]):
        power = (integer_width - indx-1)
        if int(bit):
            number += 2**power
    return sign * number