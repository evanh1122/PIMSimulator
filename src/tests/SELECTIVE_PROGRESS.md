# Selective progress: functional implementation and tests

`PIMRank` implements selective instruction progress with one shared CRF and
private control state per block. The two `SelectiveProgressBaselineFixture`
tests exercise default shared execution and explicit opt-in. The nineteen
`SelectiveProgressFixture` tests exercise the production selective interpreter;
none are feature-pending skips. Storage must be enabled for these tests.

```bash
scons NO_LIBRARY=1 NO_STORAGE=0
./sim --gtest_filter='SelectiveProgressBaselineFixture.*:SelectiveProgressFixture.*'
./sim --gtest_filter='PIMControlFixture.*:CompletionMaskingFixture.*:PIMKernelFixture.*'
```

## Difference from completion masking

Completion masking suppresses MAC arithmetic after a per-block budget runs out,
but all blocks follow one program counter. Selective progress lets a chosen
block or compatible group execute while other blocks retain their positions in
the instruction stream. Selection gates all interpreted instructions, including
stores; completion masking continues to gate only MAC arithmetic.

The implementation keeps one shared CRF and per-block state containing the PC,
last jump index, remaining jump count, last repeat index, remaining repeat count,
and exit status. Different programs per block and work reassignment are later
extensions, not requirements of this first design.

## PIMRank interface

The public methods are:

```cpp
void enableSelectiveProgress();
void doPIMSelected(BusPacket* packet, const std::vector<unsigned>& blocks);
unsigned getBlockPC(unsigned pb) const;
bool isBlockExited(unsigned pb) const;
bool allBlocksExited() const;
```

The tests call these methods directly and assert real results, counters, and PCs.

Enable the extension before starting a kernel. Each PIM-mode entry resets all
private execution states along with the existing completion budgets/counters.
With selective progress disabled by default, existing `doPIM()` callers retain
the original shared-control behavior.

Enabling is idempotent: repeated calls do not rewind divergent block positions.
On the first call, private states copy the shared position; normally enable
before kernel entry, which resets all states to instruction zero. In selective
mode, ordinary `doPIM()` calls advance all live blocks using their private states.
Thus the existing `Rank::sendToBank()` dispatch remains valid without a separate
rank-routing branch. Selective mode stays enabled for this rank instance.

Typical functional use (after loading operands and programming the CRF):

```cpp
pim->enableSelectiveProgress();  // Before the kernel-start control transaction.
// Enter PIM mode through controlPIM(), as existing kernels do.
pim->doPIMSelected(&chunk0Packet, {0});  // Advance block 0 only.
// Reload the vector chunk matching the next packet before each operation.
pim->doPIMSelected(&chunk1Packet, {0});  // Block 0 continues; block 1 is paused.
pim->doPIMSelected(&chunk0Packet, {1});  // Block 1 starts at its own instruction 0.
```

This direct API does not encode selection into timed memory transactions.

`doPIMSelected()` has this functional contract:

- Check the entire block list before executing anything. Invalid IDs throw
  `std::out_of_range`; duplicate IDs throw `std::invalid_argument`.
- The API requires selective mode and active PIM operation, otherwise it throws
  `std::logic_error`. Null packets and packet types other than READ/WRITE throw
  `std::invalid_argument`.
- An empty selection is a no-op. Selected blocks that already exited are ignored.
- Each selected live block executes at most one non-JUMP instruction per trigger,
  using its own PC and repeat/jump state. Preserve existing zero-cycle JUMP
  behavior and eight-trigger auto-repeat semantics.
- `getBlockPC()` returns the next instruction index; during auto-repeat it can
  remain on the same CRF entry across several triggers. An exited block retains
  its EXIT index.
- Unselected blocks retain PCs, loop/repeat state, accumulators, MAC budgets,
  executed/skipped counters, exit state, and bank results. Their MACs are not
  counted as skipped: a pause is different from completion masking.
- A selected MAC with zero remaining budget advances instruction state but
  suppresses arithmetic and increments its skipped count. Other selected
  instructions, including result stores, execute normally.
- The test CRF ends with a write-triggered NOP store followed by EXIT. A block
  becomes exited when its next instruction is EXIT, matching the existing
  interpreter's lookahead behavior. MAC completion alone is not program exit.
- `isToggleCond()` must remain true for valid PIM triggers while any block is
  still live. One block exiting cannot end the rank. It becomes false once all
  blocks exit; result stores must occur before that point.

These tests use one data operand from a bank per instruction and explicit packet
addresses. They do not assume free independently addressed memory accesses.

## Operand delivery and selection

One selected trigger still has one row, column, bank parity, and vector load.
The caller groups blocks only when those operands and packet type are compatible.
If block 0 needs chunk 2 and block 1 needs chunk 0, issue separate triggers with
the corresponding addresses and reload the matching input-vector chunks. Do not
infer an operand address from a PC alone: jumps and auto-repeat break that mapping.

The existing register-upload path broadcasts to every block. The test driver
deliberately retains that behavior: explicit host uploads may overwrite a paused
block's GRF_A input registers. Its accumulator and instruction/control state stay
paused. Reload the correct vector chunk before resuming it. The tests check this
by pausing and resuming blocks at different input positions.

## Implementation map

1. `PIMRank.h`: `ExecutionState` holds PC, jump/repeat bookkeeping and exit status.
   `sharedState_` supports default execution; `blockStates_` supports selection.
2. `controlPIM()` resets both shared/private states at kernel entry.
3. `executePIM()` advances an explicit state for one data instruction plus any
   zero-cycle jumps. Shared and selective modes use this same interpreter.
4. `doPIMSelected()` validates selection before execution and advances only
   selected live blocks. Each block gets packet scratch initialized from the
   original request, preventing scratch data from one block feeding another.
5. `executeBlockInstruction()` shares MAC masking/accounting across both modes
   and calls the existing `doPIMBlock()` for arithmetic and transfers.
6. `isToggleCond()` derives global completion from `allBlocksExited()`, so an
   early-exiting block cannot divert later triggers to HAB copies.

PC and exit getters validate block IDs. The interpreter rejects PCs outside the
32-entry CRF with `std::out_of_range` rather than accessing outside the array.

For performance experiments, carry selection through the real transaction and
BusPacket path, or model ordered writes to a selection register. Define that
encoding, selection overhead, bank timing, vector reloads, and result transfers
before measuring cycles. Directly changing selection in host memory between
queued transactions is not a timed control interface.

## What the tests establish

The shared fixture in `PIMControlTestFixture.h` programs the real CRF and bank
storage for a two-row block-sparse SpMV workload. The tests check single-block
selection in both directions, groups, empty/invalid selections, pause/resume with
matching operands, separate auto-repeat/jump state, per-block/global exit, masked
stores, budget preservation, and restart. Additional cases exercise ordinary
`doPIM()` and rank dispatch after one block exits, selection of non-MAC arithmetic,
idempotent enabling, and resetting partially executed jump/repeat loops. Expected
arithmetic comes from a CPU reference, not a second interpreter inside the tests.

The functional fixtures bypass the memory controller and **do not measure
cycles**. Selective progress can remove synchronization barriers only if the
workload has useful independent work ready and its operands can be delivered.
Serializing different block addresses can add commands and vector reloads. A
speedup is a hypothesis to test with a legal timed schedule, not a guaranteed
consequence of private PCs. No compaction, gather extension, or work reassignment
is implemented or assumed here.
