#include "tests/PIMControlTestFixture.h"

namespace
{
using namespace PIMControlTests;

class CompletionMaskingFixture : public PIMControlFixture
{
};

TEST_F(PIMControlFixture, block_sparse_spmv_matches_cpu_reference)
{
    run(makeWorkload({{2, 5}}, 8), false);
}

TEST_F(PIMControlFixture, zero_chunk_is_followed_by_useful_work)
{
    auto work = makeWorkload({{3, 3}}, 3);
    work.weights[0][1].fill(0);
    run(work, false);
}

TEST_F(PIMControlFixture, zero_lanes_do_not_make_the_whole_vector_zero)
{
    auto work = makeWorkload({{2, 2}}, 2);
    for (unsigned lane = 0; lane < kLanes; lane += 2) work.weights[0][0][lane] = 0;
    run(work, false);
}

TEST_F(PIMControlFixture, zero_mac_result_must_overwrite_previous_accumulator)
{
    auto work = makeWorkload({{2, 2}}, 2);
    work.input[0].fill(1);
    work.input[1].fill(1);
    work.weights[0][0].fill(1);
    work.weights[0][1].fill(-1);
    run(work, false);  // Accumulator transitions 0 -> 1 -> 0, and must be stored as 0.
}

TEST_F(CompletionMaskingFixture, unequal_work_preserves_results_and_counts_avoided_macs)
{
    const auto work = makeWorkload({{2, 5}}, 8);
    configureMask(work);
    run(work, true);  // 7 executed + 9 skipped block MACs; still 8 compute triggers.
}

TEST_F(CompletionMaskingFixture, balanced_work_masks_nothing)
{
    const auto work = makeWorkload({{4, 4}}, 4);
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, empty_first_block_does_not_stop_later_block)
{
    const auto work = makeWorkload({{0, 4}}, 8);
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, all_empty_blocks_still_store_results_and_reach_exit)
{
    const auto work = makeWorkload({{0, 0}}, 8);
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, completed_block_ignores_nonzero_padding)
{
    // Unused storage is deliberately nonzero. Only declared work may accumulate.
    const auto work = makeWorkload({{2, 5}}, 8, true);
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, zero_chunk_does_not_mark_block_complete)
{
    auto work = makeWorkload({{3, 5}}, 8);
    work.weights[0][1].fill(0);
    configureMask(work);
    run(work, true);  // The zero chunk consumes one scheduled MAC, not completion.
}

TEST_F(CompletionMaskingFixture, mixed_zero_lanes_still_count_as_one_vector_mac)
{
    auto work = makeWorkload({{2, 5}}, 8);
    for (unsigned lane = 0; lane < kLanes; lane += 2) work.weights[0][0][lane] = 0;
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, counts_executions_of_auto_repeated_instruction)
{
    const auto work = makeWorkload({{2, 5}}, 8);
    configureMask(work);
    run(work, true, Program::AutoRepeat);
}

TEST_F(CompletionMaskingFixture, counts_executions_across_jumps)
{
    const auto work = makeWorkload({{2, 5}}, 5);
    configureMask(work);
    run(work, true, Program::JumpRepeat);
}

TEST_F(CompletionMaskingFixture, reconfiguration_resets_completion_and_statistics)
{
    const auto first = makeWorkload({{1, 3}}, 4);
    configureMask(first);
    run(first, true);
    const auto second = makeWorkload({{3, 0}}, 4);
    configureMask(second);
    run(second, true);
}

TEST_F(CompletionMaskingFixture, disabling_mask_restores_unmasked_execution)
{
    const auto work = makeWorkload({{1, 0}}, 4, true);
    configureMask(work);
    pim_->disableCompletionMask();
    run(work, false);  // Unmasked execution processes all chunks, including padding.
}

TEST_F(CompletionMaskingFixture, discovers_last_useful_chunk_without_compaction)
{
    auto work = makeWorkload({{0, 0}}, 8);
    work.weights[0][0].fill(1);
    work.weights[0][3][15] = 2;  // One nonzero lane is enough to require chunk 3.
    work.validMacs = discoverMacBudgets(work);
    EXPECT_EQ(4u, work.validMacs[0]);  // Internal zero chunks 1 and 2 still execute.
    EXPECT_EQ(0u, work.validMacs[1]);
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, useful_final_chunk_requires_entire_schedule)
{
    auto work = makeWorkload({{8, 1}}, 8);
    work.validMacs = discoverMacBudgets(work);
    EXPECT_EQ(8u, work.validMacs[0]);
    EXPECT_EQ(1u, work.validMacs[1]);
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, zero_input_does_not_shorten_matrix_work_budget)
{
    auto work = makeWorkload({{2, 5}}, 8);
    for (auto& chunk : work.input) chunk.fill(0);
    work.validMacs = discoverMacBudgets(work);
    EXPECT_EQ(2u, work.validMacs[0]);
    EXPECT_EQ(5u, work.validMacs[1]);
    configureMask(work);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, kernel_restart_reloads_budgets_without_reconfiguration)
{
    const auto work = makeWorkload({{2, 5}}, 8);
    configureMask(work);
    run(work, true);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, invalid_budget_counts_leave_configuration_unchanged)
{
    const auto work = makeWorkload({{2, 5}}, 8);
    configureMask(work);
    EXPECT_THROW(pim_->configureCompletionMask({}), std::invalid_argument);
    EXPECT_THROW(pim_->configureCompletionMask({2}), std::invalid_argument);
    EXPECT_THROW(pim_->configureCompletionMask({2, 5, 8}), std::invalid_argument);
    run(work, true);
}

TEST_F(CompletionMaskingFixture, disabling_mask_mid_kernel_resumes_mac_execution)
{
    const auto work = makeWorkload({{0, 0}}, 2, true);
    configureMask(work);
    prepare(work);
    step(work, 0);
    expectAccumulators(work, {{0, 0}});

    pim_->disableCompletionMask();
    step(work, 1);
    auto expectedWork = work;
    for (auto& row : expectedWork.weights) row[0].fill(0);  // First trigger was masked.
    expectAccumulators(expectedWork, {{2, 2}});
    storeAndCheck(expectedWork, {{2, 2}});
    for (unsigned pb = 0; pb < kBlocks; ++pb)
    {
        EXPECT_EQ(1u, pim_->getExecutedMacs(pb));
        EXPECT_EQ(1u, pim_->getSkippedMacs(pb));
    }
}
}  // namespace
