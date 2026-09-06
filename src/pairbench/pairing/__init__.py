"""Cross-view pairing strategies.

- greedy_index: Z-level grouping + index-order pairing with twist/length/tilt checks.
- greedy_index_nolen: greedy_index without the nominal-length check.
- greedy_index_order: greedy_index without any check (instance order alone).
- hungarian: global assignment on folded angle, YZ distance and Z-level cost.
- hungarian_aligned: RANSAC translation compensation, then hungarian.
- monotone_aligned: same cost, assignment constrained to preserve Y order.
- hybrid_aligned: monotone_aligned plus evidence-gated crossing hypotheses.

hybrid_aligned is the default strategy. `hypotheses` enumerates and costs
candidate translations explicitly.
"""

from functools import partial

from .common import PairingResult, fold_angle, group_by_z_level
from .greedy_index import pair_greedy_index
from .hungarian import pair_hungarian
from .hungarian_aligned import pair_hungarian_aligned
from .hybrid import pair_hybrid_aligned
from .monotone import pair_monotone_aligned

STRATEGIES = {
    "greedy_index": pair_greedy_index,
    "greedy_index_nolen": partial(pair_greedy_index, check_length=False),
    "greedy_index_order": partial(pair_greedy_index, check_length=False,
                                  check_twist=False, check_tilt=False),
    "hungarian": pair_hungarian,
    "hungarian_aligned": pair_hungarian_aligned,
    "monotone_aligned": pair_monotone_aligned,
    "hybrid_aligned": pair_hybrid_aligned,
}
