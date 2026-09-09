# Can the objective tell the true correspondence from a shifted one?

Detections: `B1 + B2 rescued`. 102 scenes with at least two resolved claims; 102 of them have at least one rival hypothesis, i.e. all of them.

For each scene the annotated correspondence is scored as a hypothesis under the order-constrained objective the aligned arms minimize, and compared against its rivals: the order-constrained solutions under every translation the scene's own detections propose, plus the members pi_k of the shift family for k in [-2, -1, 1, 2], built level by level. `gap = cost(cheapest rival) - cost(true)`; `margin_shift1` is the same difference against the cheaper of pi_-1 and pi_+1 alone.

## Result

- median gap **+7.074**, IQR [+6.153, +9.009]
- median margin against a one-bar shift **+7.307**, IQR [+6.495, +11.690], minimum +0.840; negative on **0 of 102** scenes
- the objective ranks a **different** hypothesis first on **2 of 102** scenes
- what the cheapest rival is, over all scenes: other 12, shift +1 35, shift +2 1, shift -1 43, shift -2 5, subset of truth 6

A negative margin against a one-bar shift is the aliasing under test. A shift rival is a wrong association; a subset of the truth only leaves a true pair unmatched because its cost exceeds the unmatch cost.

## Scenes where the objective prefers a different hypothesis

| scene | claims | rival | cost(true) | cost(rival) | gap |
|---|---|---|---|---|---|
| l-profile/crossed-07 | 3 | subset of truth | 8.302 | 7.967 | -0.335 |
| l-profile/crossed-08 | 3 | subset of truth | 8.116 | 7.967 | -0.149 |

Full per-scene values: `cost_gap_pool.csv`.

## `hybrid_aligned`'s wrong scenes on the full detection set (`B1 confirmed`)

The comparison above is restricted to witnessed detections. Here every scene `hybrid_aligned` gets wholly wrong (at least two resolved claims, none correct) is re-costed on the full detection set: `full_gap = cost(arm output) - cost(annotated pairs)`, every unwitnessed detection left unmatched in the latter. `spare` lists the detections without an annotated partner, by index along Y on each side.

| scene | claims | detections (1/2) | spare 1 | spare 2 | pairs in output | counting | cost(output) | cost(truth) | full_gap |
|---|---|---|---|---|---|---|---|---|---|
| l-profile-reshoot/overlap-08 | 4 | 4/5 | — | [0] | 4 | tie | 5.53 | 6.13 | -0.60 |
| l-profile/overlap-04 | 3 | 4/4 | [0] | [3] | 4 | shift pairs more | 2.00 | 9.35 | -7.36 |
| l-profile/overlap-10 | 3 | 4/4 | [0] | [3] | 4 | shift pairs more | 1.95 | 8.55 | -6.60 |
| l-profile/single-04 | 2 | 3/2 | [2] | — | 2 | tie | 3.84 | 3.98 | -0.14 |
| square-profile/single-26 | 7 | 7/8 | — | [0] | 7 | tie | 4.00 | 4.02 | -0.02 |

On **5 of 5** of these scenes the objective, costed on everything the arm saw, prefers the arm's wrong output to the annotated pairs.

Every scene with at least two resolved claims, by whether all of its detections on both sides carry an annotated partner:

| every detection witnessed | scenes | wholly wrong |
|---|---|---|
| yes | 59 | 0 |
| no | 43 | 5 |

A detection without an annotated partner may be a bar the other view truncates, a false detection or an unclaimed bar; this table does not tell those apart.
