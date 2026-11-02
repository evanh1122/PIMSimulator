# Completion masking without compaction

Run from the repository root, with storage enabled:

```bash
scons NO_LIBRARY=1 NO_STORAGE=0
./sim --gtest_filter='PIMControlFixture.*:CompletionMaskingFixture.*'
./sim --gtest_filter='PIMKernelFixture.*'
```

The four `PIMControlFixture` tests exercise unmasked behavior. The seventeen
`CompletionMaskingFixture` tests exercise the production masking implementation
in `PIMRank`. All tests run with storage enabled; there are no feature-pending
skips. The fixtures skip in a `NO_STORAGE=1` build because they check arithmetic
and bank contents.

## Workload and reference

Two PIM blocks each own one matrix row. Each compute trigger processes the next
16 columns using a shared input-vector chunk and block-local bank weights:

```text
GRF_B[0][lane] += vector_chunk[lane] * local_weight_chunk[lane]
```

The test driver loads weights into each block's even bank, broadcasts each
matching vector chunk, programs the real CRF, and calls `PIMRank::doPIM()`.
It checks accumulators after every trigger, stores them through the existing
write-triggered NOP path, and checks bank contents and the resulting dot products
against a separate CPU loop. Small integer operands are exactly representable
throughout these tests in FP16; exact comparisons are intentional. More general
floating-point workloads will need documented absolute/relative tolerances.

These are **functional tests that bypass controller scheduling and timing**.
Calling `doPIM()` eight times is eight triggers, not a measurement of eight DRAM
cycles. These tests cannot establish speedup or energy savings. A later timed
benchmark must use the transaction/controller path and charge descriptor loading,
vector/result traffic, and selection overhead.

## Production interface

`PIMRank` exposes these methods:

```cpp
void configureCompletionMask(const std::vector<unsigned>& macCounts);
void disableCompletionMask();
uint64_t getExecutedMacs(unsigned pb) const;
uint64_t getSkippedMacs(unsigned pb) const;
```

Configure before entering PIM mode, with one budget per block. Incorrect vector
lengths throw `std::invalid_argument` without changing the existing configuration.
The getters reject invalid block IDs with `std::out_of_range`.

For example, in the two-block fixture:

```cpp
pim->configureCompletionMask({2, 5});
// Start a kernel and issue eight MAC triggers, then store results.
// Block 0: 2 executed, 6 skipped. Block 1: 5 executed, 3 skipped.
```

The test contract is:

- Masking is disabled by default, preserving existing kernels.
- One budget unit is one vector MAC execution on one block, including each
  execution of a repeated CRF instruction. It is not a scalar lane or CRF entry.
- Counts are per block and describe the required **prefix** of compute triggers
  in this workload. A zero-valued chunk inside that prefix still consumes one
  MAC. An empty block has a zero budget from the start.
- Configuration supplies one count per block and starts a fresh accounting
  interval. Counts configured before PIM-mode entry must survive that entry.
- Every MAC trigger records one executed or skipped MAC per block. Register
  broadcasts, jumps, NOPs, stores, and EXIT do not consume MAC budgets/counters.
- Reaching zero suppresses subsequent MACs for that block while preserving its
  accumulator. It does not end the shared program or prevent result stores.
- Configuration and each PIM-mode entry reload the saved budgets and reset both
  counters, following the existing shared-PC reset on mode entry.
- Executed MACs are also counted when masking is disabled. Disabling the mask
  immediately restores unmasked execution, even mid-kernel, and retains the
  current counters until configuration or the next PIM-mode entry resets them.

The configuration method is a functional test seam. It does not model the time
or bandwidth needed to transfer work descriptors. Choose and implement a modeled
control transaction before using the extension in performance experiments. The
existing dense GEMV kernel remains unmasked by default; its layout and trigger
generation are unchanged.

## Finding budgets without moving data

`discoverMacBudgets()` in `CompletionMaskingTestCases.cpp` scans matrix chunks in
compute-trigger order. For each block it returns one plus the index of the last
chunk containing any nonzero FP16 weight, or zero for an all-zero assignment.
It quantizes test weights to FP16 just as the preload path does, and does not
depend on the input vector's current values.

```text
chunks:   useful, zero, zero, useful, zero, zero, zero, zero
budget:   4
behavior: execute first four MACs; mask the remaining four
```

The helper runs on the host side before simulation, and does not reorder weights
or vector operands. Use it on ordinary matrix data; keep explicit budgets for
descriptor tests with nonzero sentinels outside the declared valid region.

For the existing multirow GEMV kernel, derive budgets using its actual bank and
output-register mapping: `preloadGemv()` determines placement, and
`executeGemv()`/`computeGemv()` determine the sequence of MAC operands. That kernel
processes even-bank tiles before odd-bank tiles, so a simple last column index
is not its execution budget. Automatic discovery is currently implemented only
for the small test workload.

## Cases

| Fixture/case | Purpose |
| --- | --- |
| Baseline block-sparse SpMV | Validate data placement, vector delivery, and CPU reference. |
| Zero chunk followed by useful work | A zero contribution does not imply completion. |
| Mixed zero/nonzero lanes | One zero lane does not make an entire MAC unnecessary. |
| MAC result becomes zero | Cancellation must overwrite an earlier nonzero accumulator. |
| Unequal budgets `[2,5]`, 8 triggers | Expect 7 executed and 9 skipped block MACs. |
| Balanced budgets `[4,4]` | Expect no skipped MACs. |
| Empty first block | Detect accidentally breaking out before other blocks execute. |
| All blocks empty | Preserve zero results, store them, and reach shared EXIT. |
| Nonzero unused padding | Prove completion state, rather than scanning values, gates work. |
| Internal zero chunk / mixed lanes with masking | Budgets count scheduled vector MACs, not nonzeros. |
| Auto-repeat and JUMP | Count actual MAC executions rather than static instructions. |
| Reconfigure/restart | Avoid stale completion and accumulated statistics. |
| Disable masking | Restore the original execution behavior. |
| Discover budgets with internal zeros / a useful final chunk | Retain original chunk positions and find the true end of required work. |
| All-zero input vector | Discover budgets from matrix weights, independently of current input values. |
| Restart without reconfiguration | Reload the saved budgets and reset statistics. |
| Invalid budget vector lengths | Reject bad descriptors without losing the valid configuration. |
| Disable masking mid-kernel | Resume MACs immediately while retaining accounting. |

The padding test deliberately puts nonzero sentinel data outside each row's
declared valid prefix. That region is unused storage, not an omitted valid
matrix contribution. Ordinary baseline sparse cases use zero padding. Internal
sparse holes continue to execute and consume budget; only the trailing suffix
is masked. Skipping those internal holes would require a validity schedule or
operand-indexing design; merely counting nonzeros does not provide one.
