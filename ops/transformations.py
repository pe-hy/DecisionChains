"""Transformation functions for BranchingTokens.

Each transformation:
- Takes a vector (list of ints), modulo k, and a trace list
- Returns a new vector (same length)
- Appends its trace string to the trace list
- All values are mod k after operation
"""

from typing import Callable, List, Union


def _format_vec(vec: List[int]) -> str:
    """Format vector as [ a , b , c ] for whitespace tokenization."""
    return "[ " + " , ".join(str(x) for x in vec) + " ]"


def reverse(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Reverse the vector."""
    result = vec[::-1]
    trace.append(f"reverse R {_format_vec(result)}")
    return result


def sort_asc(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Sort ascending."""
    result = sorted(vec)
    trace.append(f"sort_asc R {_format_vec(result)}")
    return result


def sort_desc(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Sort descending."""
    result = sorted(vec, reverse=True)
    trace.append(f"sort_desc R {_format_vec(result)}")
    return result


def cumsum(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative sum with mod k."""
    result = []
    running = 0
    ops = []
    for i, x in enumerate(vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running + x) % k
            ops.append(f"{running} + {x} = {new_val}")
            result.append(new_val)
        running = result[-1]
    trace.append(f"cumsum : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def cumsum_reverse(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative sum from right with mod k."""
    reversed_vec = vec[::-1]
    result = []
    running = 0
    ops = []
    for i, x in enumerate(reversed_vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running + x) % k
            ops.append(f"{running} + {x} = {new_val}")
            result.append(new_val)
        running = result[-1]
    result = result[::-1]
    trace.append(f"cumsum_reverse : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def add_1(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 1 to all elements mod k."""
    result = [(x + 1) % k for x in vec]
    ops = [f"{x} + 1 = {r}" for x, r in zip(vec, result)]
    trace.append(f"add_1 : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def add_2(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 2 to all elements mod k."""
    result = [(x + 2) % k for x in vec]
    ops = [f"{x} + 2 = {r}" for x, r in zip(vec, result)]
    trace.append(f"add_2 : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def add_3(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add 3 to all elements mod k."""
    result = [(x + 3) % k for x in vec]
    ops = [f"{x} + 3 = {r}" for x, r in zip(vec, result)]
    trace.append(f"add_3 : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def diff(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Pairwise differences. First element unchanged, rest are vec[i] - vec[i-1] mod k."""
    result = [vec[0]]
    ops = [str(vec[0])]
    for i in range(1, len(vec)):
        d = (vec[i] - vec[i - 1]) % k
        ops.append(f"{vec[i]} - {vec[i-1]} = {d}")
        result.append(d)
    trace.append(f"diff : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def swap_pairs(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Swap adjacent pairs. Odd-length vectors keep last element in place."""
    result = list(vec)
    swaps = []
    for i in range(0, len(vec) - 1, 2):
        result[i], result[i + 1] = result[i + 1], result[i]
        swaps.append(f"( {vec[i]} , {vec[i+1]} ) -> ( {result[i]} , {result[i+1]} )")
    if len(vec) % 2 == 1:
        swaps.append(f"{vec[-1]} -> stays")
    trace.append(f"swap_pairs : {' , '.join(swaps)} R {_format_vec(result)}")
    return result


def rotate_left(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Rotate left by 1."""
    result = vec[1:] + vec[:1]
    trace.append(f"rotate_left R {_format_vec(result)}")
    return result


def rotate_right(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Rotate right by 1."""
    result = vec[-1:] + vec[:-1]
    trace.append(f"rotate_right R {_format_vec(result)}")
    return result


def negate(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Negate: x -> k - x."""
    result = [(k - x) % k for x in vec]
    ops = [f"{k} - {x} = {r}" for x, r in zip(vec, result)]
    trace.append(f"negate : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def double(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Double all elements mod k."""
    result = [(2 * x) % k for x in vec]
    ops = [f"2 * {x} = {r}" for x, r in zip(vec, result)]
    trace.append(f"double : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def square(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Square all elements mod k."""
    result = [(x * x) % k for x in vec]
    ops = [f"{x} ^ 2 = {r}" for x, r in zip(vec, result)]
    trace.append(f"square : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def min_prefix(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Running minimum."""
    result = []
    running_min = float('inf')
    ops = []
    for x in vec:
        running_min = min(running_min, x)
        result.append(running_min)
        ops.append(f"min = {running_min}")
    trace.append(f"min_prefix : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def scale_by_first(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Multiply all elements by first element mod k."""
    result = [(x * vec[0]) % k for x in vec]
    ops = [f"{x} * {vec[0]} = {r}" for x, r in zip(vec, result)]
    trace.append(f"scale_by_first : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def scale_by_last(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Multiply all elements by last element mod k."""
    result = [(x * vec[-1]) % k for x in vec]
    ops = [f"{x} * {vec[-1]} = {r}" for x, r in zip(vec, result)]
    trace.append(f"scale_by_last : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def add_first_to_all(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add first element to all elements mod k."""
    result = [(x + vec[0]) % k for x in vec]
    ops = [f"{x} + {vec[0]} = {r}" for x, r in zip(vec, result)]
    trace.append(f"add_first_to_all : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def add_last_to_all(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add last element to all elements mod k."""
    result = [(x + vec[-1]) % k for x in vec]
    ops = [f"{x} + {vec[-1]} = {r}" for x, r in zip(vec, result)]
    trace.append(f"add_last_to_all : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def prepend_sum(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Prepend sum of all elements, drop last."""
    s = sum(vec) % k
    result = [s] + vec[:-1]
    trace.append(f"prepend_sum : sum = {s} R {_format_vec(result)}")
    return result


def append_sum(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Append sum of all elements, drop first."""
    s = sum(vec) % k
    result = vec[1:] + [s]
    trace.append(f"append_sum : sum = {s} R {_format_vec(result)}")
    return result


def diff_from_first(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Subtract first element from all mod k."""
    result = [(x - vec[0]) % k for x in vec]
    ops = [f"{x} - {vec[0]} = {r}" for x, r in zip(vec, result)]
    trace.append(f"diff_from_first : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def diff_from_last(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Subtract last element from all mod k."""
    result = [(x - vec[-1]) % k for x in vec]
    ops = [f"{x} - {vec[-1]} = {r}" for x, r in zip(vec, result)]
    trace.append(f"diff_from_last : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def prefix_product(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative product mod k."""
    result = []
    running = 1
    ops = []
    for i, x in enumerate(vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running * x) % k
            ops.append(f"{running} * {x} = {new_val}")
            result.append(new_val)
        running = result[-1]
    trace.append(f"prefix_product : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def interleave_sum_diff(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Replace pairs with (sum, diff). Odd-length keeps last."""
    result = []
    ops = []
    for i in range(0, len(vec) - 1, 2):
        s = (vec[i] + vec[i + 1]) % k
        d = (vec[i] - vec[i + 1]) % k
        result.extend([s, d])
        ops.append(f"({vec[i]} + {vec[i+1]}, {vec[i]} - {vec[i+1]}) = ({s}, {d})")
    if len(vec) % 2 == 1:
        result.append(vec[-1])
        ops.append(f"{vec[-1]} stays")
    trace.append(f"interleave_sum_diff : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def sliding_sum(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Sliding window sum (circular)."""
    result = []
    ops = []
    for i in range(len(vec)):
        s = (vec[i] + vec[(i + 1) % len(vec)]) % k
        result.append(s)
        ops.append(f"{vec[i]} + {vec[(i+1) % len(vec)]} = {s}")
    trace.append(f"sliding_sum : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def position_multiply(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Multiply each element by its position mod k."""
    result = [(x * i) % k for i, x in enumerate(vec)]
    ops = [f"{x} * {i} = {r}" for i, (x, r) in enumerate(zip(vec, result))]
    trace.append(f"position_multiply : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def position_add(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add position to each element mod k."""
    result = [(x + i) % k for i, x in enumerate(vec)]
    ops = [f"{x} + {i} = {r}" for i, (x, r) in enumerate(zip(vec, result))]
    trace.append(f"position_add : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def alternating_add_sub(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Add position to even indices, subtract from odd mod k."""
    result = [(x + i) % k if i % 2 == 0 else (x - i) % k for i, x in enumerate(vec)]
    ops = [f"{vec[i]} + {i} = {result[i]}" if i % 2 == 0 else f"{vec[i]} - {i} = {result[i]}" for i in range(len(vec))]
    trace.append(f"alternating_add_sub : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def conditional_double(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Double elements >= k/2, keep others mod k."""
    threshold = k // 2
    result = [(2 * x) % k if x >= threshold else x for x in vec]
    ops = [f"2 * {x} = {r}" if x >= threshold else str(x) for x, r in zip(vec, result)]
    trace.append(f"conditional_double : {' , '.join(ops)} R {_format_vec(result)}")
    return result


def reverse_cumsum(vec: List[int], k: int, trace: List[str]) -> List[int]:
    """Cumulative sum but computed from right to left, result reversed."""
    reversed_vec = vec[::-1]
    result = []
    running = 0
    ops = []
    for i, x in enumerate(reversed_vec):
        if i == 0:
            result.append(x % k)
            ops.append(str(x % k))
        else:
            new_val = (running + x) % k
            ops.append(f"{running} + {x} = {new_val}")
            result.append(new_val)
        running = result[-1]
    result = result[::-1]
    trace.append(f"reverse_cumsum : {' , '.join(ops[::-1])} R {_format_vec(result)}")
    return result


TransformFn = Callable[[List[int], int, List[str]], List[int]]

TRANSFORMATIONS: dict[str, TransformFn] = {
    "reverse": reverse,
    "sort_asc": sort_asc,
    "sort_desc": sort_desc,
    "cumsum": cumsum,
    "cumsum_reverse": cumsum_reverse,
    "add_1": add_1,
    "add_2": add_2,
    "add_3": add_3,
    "diff": diff,
    "swap_pairs": swap_pairs,
    "rotate_left": rotate_left,
    "rotate_right": rotate_right,
    "negate": negate,
    "double": double,
    "square": square,
    "min_prefix": min_prefix,
    "scale_by_first": scale_by_first,
    "scale_by_last": scale_by_last,
    "add_first_to_all": add_first_to_all,
    "add_last_to_all": add_last_to_all,
    "prepend_sum": prepend_sum,
    "append_sum": append_sum,
    "diff_from_first": diff_from_first,
    "diff_from_last": diff_from_last,
    "prefix_product": prefix_product,
    "interleave_sum_diff": interleave_sum_diff,
    "sliding_sum": sliding_sum,
    "position_multiply": position_multiply,
    "position_add": position_add,
    "alternating_add_sub": alternating_add_sub,
    "conditional_double": conditional_double,
    "reverse_cumsum": reverse_cumsum,
}


def get_transformation(key: Union[str, int]) -> TransformFn:
    """Get transformation by name or index."""
    if isinstance(key, int):
        name = list(TRANSFORMATIONS.keys())[key]
        return TRANSFORMATIONS[name]
    return TRANSFORMATIONS[key]