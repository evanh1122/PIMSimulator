// Functional selective-progress tests against the production PIMRank interpreter.
#include "tests/PIMControlTestFixture.h"

namespace
{
using namespace PIMControlTests;

class SelectiveProgressBaselineFixture : public PIMControlFixture {};

TEST_F(SelectiveProgressBaselineFixture, legacy_trigger_advances_both_blocks)
{
    const auto work = makeWorkload({{3, 3}}, 3);
    prepare(work);
    step(work, 0);
    expectAccumulators(work, {{1, 1}});
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(1u, pim_->getExecutedMacs(1));
}

TEST_F(SelectiveProgressBaselineFixture, selected_api_requires_explicit_enabling)
{
    const auto work = makeWorkload({{1, 1}}, 1);
    prepare(work);
    BurstType scratch;
    BusPacket trigger(READ, 0, 0, 0, 0, 0, &scratch, log_);
    EXPECT_THROW(pim_->doPIMSelected(&trigger, {0}), std::logic_error);
    step(work, 0);
    expectAccumulators(work, {{1, 1}});
    storeAndCheck(work, {{1, 1}});
}

class SelectiveProgressFixture : public PIMControlFixture
{
  protected:
    void SetUp() override
    {
        PIMControlFixture::SetUp();
        if (IsSkipped() || HasFatalFailure()) return;
        pim_->enableSelectiveProgress();
    }

    void triggerSelected(unsigned round, const std::vector<unsigned>& blocks)
    {
        BurstType scratch;
        BusPacket trigger(READ, 0, round, 0, 0, 0, &scratch, log_);
        ASSERT_TRUE(pim_->isToggleCond(&trigger));
        pim_->doPIMSelected(&trigger, blocks);
    }

    void stepSelected(const Workload& work, unsigned round, const std::vector<unsigned>& blocks)
    {
        BurstType input;
        for (unsigned lane = 0; lane < kLanes; ++lane)
            input.fp16Data_[lane] = convertF2H(work.input[round][lane]);
        const unsigned reg = program_ == Program::AutoRepeat ? round : 0;
        BusPacket load(WRITE, 0, 0x8 + reg, config_->PIM_REG_RA, 0, 0, &input, log_);
        pim_->writeHab(&load);
        // The caller supplies the exact bank offset and matching vector chunk.
        // Selection does not secretly gather data or advance an address cursor.
        triggerSelected(round, blocks);
    }

    void storeSelected(const std::vector<unsigned>& blocks)
    {
        BurstType scratch;
        BusPacket store(WRITE, 0, 0, 2, 0, 1, &scratch, log_);
        ASSERT_TRUE(pim_->isToggleCond(&store));
        pim_->doPIMSelected(&store, blocks);
        for (unsigned pb : blocks) EXPECT_TRUE(pim_->isBlockExited(pb));
    }

    BurstType readStored(unsigned pb)
    {
        BurstType output;
        output.set(convertF2H(-456));
        BusPacket read(READ, 0, 0, 2, 0, 2 * pb + 1, &output, log_);
        rank_->banks[2 * pb + 1].read(&read);
        return output;
    }

    void expectStored(const Workload& work, unsigned pb, unsigned macs)
    {
        const auto expected = reference(work, pb, macs);
        const auto output = readStored(pb);
        for (unsigned lane = 0; lane < kLanes; ++lane)
            EXPECT_DOUBLE_EQ(expected[lane], convertH2F(output.fp16Data_[lane]))
                << "stored block=" << pb << " lane=" << lane;
    }

    void expectUnstored(unsigned pb)
    {
        const auto output = readStored(pb);
        for (unsigned lane = 0; lane < kLanes; ++lane)
            EXPECT_FLOAT_EQ(-123, convertH2F(output.fp16Data_[lane]));
    }

};

TEST_F(SelectiveProgressFixture, selecting_first_block_pauses_the_other)
{
    const auto work = makeWorkload({{3, 3}}, 3);
    prepare(work);
    stepSelected(work, 0, {0});
    expectAccumulators(work, {{1, 0}});
    EXPECT_EQ(1u, pim_->getBlockPC(0));
    EXPECT_EQ(0u, pim_->getBlockPC(1));
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(0u, pim_->getExecutedMacs(1));
    EXPECT_EQ(0u, pim_->getSkippedMacs(1));  // Paused is different from masked.
}

TEST_F(SelectiveProgressFixture, selecting_last_block_does_not_advance_earlier_block)
{
    const auto work = makeWorkload({{3, 3}}, 3);
    prepare(work);
    stepSelected(work, 0, {1});
    expectAccumulators(work, {{0, 1}});
    EXPECT_EQ(0u, pim_->getBlockPC(0));
    EXPECT_EQ(1u, pim_->getBlockPC(1));
}

TEST_F(SelectiveProgressFixture, resumed_block_uses_its_own_pc_and_matching_operands)
{
    const auto work = makeWorkload({{3, 3}}, 3);
    prepare(work);
    stepSelected(work, 0, {0});
    stepSelected(work, 1, {0});
    stepSelected(work, 0, {1});  // Reload x[0], not the x[1] just used by block 0.
    expectAccumulators(work, {{2, 1}});
    EXPECT_EQ(2u, pim_->getBlockPC(0));
    EXPECT_EQ(1u, pim_->getBlockPC(1));
    stepSelected(work, 2, {0});
    storeSelected({0});
    expectStored(work, 0, 3);
    expectUnstored(1);
    stepSelected(work, 1, {1});
    stepSelected(work, 2, {1});
    storeSelected({1});
    expectStored(work, 1, 3);
    EXPECT_TRUE(pim_->allBlocksExited());
}

TEST_F(SelectiveProgressFixture, matching_group_executes_shared_operand_schedule)
{
    const auto work = makeWorkload({{3, 3}}, 3);
    prepare(work);
    for (unsigned round = 0; round < 3; ++round) stepSelected(work, round, {0, 1});
    expectAccumulators(work, {{3, 3}});
    EXPECT_EQ(3u, pim_->getBlockPC(0));
    EXPECT_EQ(3u, pim_->getBlockPC(1));
    storeSelected({0, 1});
    for (unsigned pb = 0; pb < kBlocks; ++pb)
    {
        expectStored(work, pb, 3);
        EXPECT_EQ(3u, pim_->getExecutedMacs(pb));
        EXPECT_EQ(0u, pim_->getSkippedMacs(pb));
    }
    EXPECT_TRUE(pim_->allBlocksExited());
}

TEST_F(SelectiveProgressFixture, empty_selection_consumes_no_instructions_or_mac_budgets)
{
    const auto work = makeWorkload({{1, 3}}, 3);
    configureMask(work);
    prepare(work);
    triggerSelected(0, {});
    expectAccumulators(work, {{0, 0}});
    for (unsigned pb = 0; pb < kBlocks; ++pb)
    {
        EXPECT_EQ(0u, pim_->getBlockPC(pb));
        EXPECT_EQ(0u, pim_->getExecutedMacs(pb));
        EXPECT_EQ(0u, pim_->getSkippedMacs(pb));
        EXPECT_FALSE(pim_->isBlockExited(pb));
    }
    for (unsigned round = 0; round < 3; ++round) stepSelected(work, round, {0, 1});
    storeSelected({0, 1});
    expectStored(work, 0, 1);
    expectStored(work, 1, 3);
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(2u, pim_->getSkippedMacs(0));
    EXPECT_EQ(3u, pim_->getExecutedMacs(1));
}

TEST_F(SelectiveProgressFixture, one_block_exiting_does_not_end_rank_execution)
{
    const auto work = makeWorkload({{1, 1}}, 1);
    prepare(work);
    stepSelected(work, 0, {0});
    storeSelected({0});
    EXPECT_TRUE(pim_->isBlockExited(0));
    EXPECT_FALSE(pim_->isBlockExited(1));
    EXPECT_FALSE(pim_->allBlocksExited());
    stepSelected(work, 0, {1});
    storeSelected({1});
    expectStored(work, 0, 1);
    expectStored(work, 1, 1);
    EXPECT_TRUE(pim_->allBlocksExited());
    BurstType scratch;
    BusPacket probe(READ, 0, 0, 0, 0, 0, &scratch, log_);
    EXPECT_FALSE(pim_->isToggleCond(&probe));
}

TEST_F(SelectiveProgressFixture, selecting_exited_block_is_a_noop)
{
    const auto work = makeWorkload({{1, 1}}, 1);
    prepare(work);
    stepSelected(work, 0, {0});
    storeSelected({0});
    const unsigned exitedPC = pim_->getBlockPC(0);
    triggerSelected(0, {0});
    EXPECT_EQ(exitedPC, pim_->getBlockPC(0));
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(0u, pim_->getSkippedMacs(0));
    EXPECT_EQ(0u, pim_->getBlockPC(1));
    stepSelected(work, 0, {0, 1});  // Only the live block participates.
    expectAccumulators(work, {{1, 1}});
    storeSelected({1});
    EXPECT_TRUE(pim_->allBlocksExited());
}

TEST_F(SelectiveProgressFixture, auto_repeat_state_is_private_to_each_block)
{
    const auto work = makeWorkload({{8, 8}}, 8);
    prepare(work, Program::AutoRepeat);
    stepSelected(work, 0, {0});
    stepSelected(work, 1, {0});
    stepSelected(work, 0, {1});
    for (unsigned round = 2; round < 8; ++round) stepSelected(work, round, {0});
    EXPECT_EQ(1u, pim_->getBlockPC(0));  // Eight executions of CRF instruction 0.
    EXPECT_EQ(0u, pim_->getBlockPC(1));
    storeSelected({0});
    for (unsigned round = 1; round < 8; ++round) stepSelected(work, round, {1});
    storeSelected({1});
    for (unsigned pb = 0; pb < kBlocks; ++pb)
    {
        expectStored(work, pb, 8);
        EXPECT_EQ(8u, pim_->getExecutedMacs(pb));
    }
}

TEST_F(SelectiveProgressFixture, jump_loop_state_is_private_to_each_block)
{
    const auto work = makeWorkload({{4, 4}}, 4);
    prepare(work, Program::JumpRepeat);
    stepSelected(work, 0, {0});
    stepSelected(work, 1, {0});
    stepSelected(work, 0, {1});
    stepSelected(work, 2, {0});
    stepSelected(work, 3, {0});
    storeSelected({0});
    for (unsigned round = 1; round < 4; ++round) stepSelected(work, round, {1});
    storeSelected({1});
    for (unsigned pb = 0; pb < kBlocks; ++pb)
    {
        expectStored(work, pb, 4);
        EXPECT_EQ(4u, pim_->getExecutedMacs(pb));
    }
}

TEST_F(SelectiveProgressFixture, paused_block_retains_completion_budget_and_statistics)
{
    const auto work = makeWorkload({{1, 3}}, 3, true);
    configureMask(work);
    prepare(work);
    for (unsigned round = 0; round < 3; ++round) stepSelected(work, round, {0});
    expectAccumulators(work, {{1, 0}});
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(2u, pim_->getSkippedMacs(0));
    EXPECT_EQ(0u, pim_->getExecutedMacs(1));
    EXPECT_EQ(0u, pim_->getSkippedMacs(1));
    EXPECT_EQ(0u, pim_->getBlockPC(1));
    storeSelected({0});
    for (unsigned round = 0; round < 3; ++round) stepSelected(work, round, {1});
    storeSelected({1});
    expectStored(work, 0, 1);
    expectStored(work, 1, 3);
    EXPECT_EQ(3u, pim_->getExecutedMacs(1));
    EXPECT_EQ(0u, pim_->getSkippedMacs(1));
}

TEST_F(SelectiveProgressFixture, masked_block_still_progresses_to_its_result_store)
{
    const auto work = makeWorkload({{0, 2}}, 3);
    configureMask(work);
    prepare(work);
    for (unsigned round = 0; round < 3; ++round) stepSelected(work, round, {0, 1});
    EXPECT_FALSE(pim_->allBlocksExited());  // No result has been stored yet.
    storeSelected({0});
    expectStored(work, 0, 0);
    expectUnstored(1);
    EXPECT_FALSE(pim_->allBlocksExited());
    storeSelected({1});
    expectStored(work, 1, 2);
    EXPECT_EQ(3u, pim_->getSkippedMacs(0));
    EXPECT_EQ(1u, pim_->getSkippedMacs(1));
    EXPECT_TRUE(pim_->allBlocksExited());
}

TEST_F(SelectiveProgressFixture, kernel_restart_resets_private_progress_and_accounting)
{
    const auto work = makeWorkload({{1, 2}}, 2);
    configureMask(work);
    prepare(work);
    stepSelected(work, 0, {0});
    // Restart while block 0 is ahead; both PCs must return to the entry point.
    prepare(work);
    for (unsigned pb = 0; pb < kBlocks; ++pb)
    {
        EXPECT_EQ(0u, pim_->getBlockPC(pb));
        EXPECT_FALSE(pim_->isBlockExited(pb));
        EXPECT_EQ(0u, pim_->getExecutedMacs(pb));
        EXPECT_EQ(0u, pim_->getSkippedMacs(pb));
    }
    for (unsigned round = 0; round < 2; ++round) stepSelected(work, round, {0, 1});
    storeSelected({0, 1});
    expectStored(work, 0, 1);
    expectStored(work, 1, 2);
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(1u, pim_->getSkippedMacs(0));
    EXPECT_EQ(2u, pim_->getExecutedMacs(1));
}

TEST_F(SelectiveProgressFixture, invalid_and_duplicate_selections_are_rejected_atomically)
{
    const auto work = makeWorkload({{2, 2}}, 2);
    prepare(work);
    stepSelected(work, 0, {0});
    // Validate the whole selection before allowing its first valid block to run.
    EXPECT_THROW(triggerSelected(0, {1, kBlocks}), std::out_of_range);
    EXPECT_THROW(triggerSelected(1, {0, 0}), std::invalid_argument);
    expectAccumulators(work, {{1, 0}});
    EXPECT_EQ(1u, pim_->getBlockPC(0));
    EXPECT_EQ(0u, pim_->getBlockPC(1));
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(0u, pim_->getExecutedMacs(1));
}

TEST_F(SelectiveProgressFixture, internal_zero_chunk_does_not_end_selected_block)
{
    auto work = makeWorkload({{3, 3}}, 3);
    work.weights[0][1].fill(0);
    prepare(work);
    for (unsigned round = 0; round < 3; ++round) stepSelected(work, round, {0});
    expectAccumulators(work, {{3, 0}});
    EXPECT_EQ(3u, pim_->getExecutedMacs(0));
    EXPECT_FALSE(pim_->isBlockExited(0));  // The pending result store must execute.
    storeSelected({0});
    expectStored(work, 0, 3);
    expectUnstored(1);
    EXPECT_FALSE(pim_->allBlocksExited());
}

TEST_F(SelectiveProgressFixture, ordinary_trigger_advances_all_live_private_states)
{
    const auto work = makeWorkload({{1, 1}}, 1);
    prepare(work);
    stepSelected(work, 0, {0});
    storeSelected({0});
    step(work, 0);  // Existing doPIM() callers select every live block in this mode.
    expectAccumulators(work, {{1, 1}});
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(1u, pim_->getExecutedMacs(1));
    storeSelected({1});
    expectStored(work, 0, 1);
    expectStored(work, 1, 1);
    EXPECT_TRUE(pim_->allBlocksExited());
}

TEST_F(SelectiveProgressFixture, rank_dispatch_keeps_executing_after_one_block_exits)
{
    const auto work = makeWorkload({{1, 1}}, 1);
    prepare(work);
    stepSelected(work, 0, {0});
    storeSelected({0});

    // Exercise Rank's existing dispatch without claiming controller timing.
    // The vector chunk loaded above is still correct for the remaining block.
    BurstType scratch;
    rank_->sendToBank(new BusPacket(READ, 0, 0, 0, 0, 0, &scratch, log_));
    expectAccumulators(work, {{1, 1}});
    EXPECT_EQ(1u, pim_->getExecutedMacs(0));
    EXPECT_EQ(1u, pim_->getExecutedMacs(1));
    ASSERT_EQ(1u, rank_->readReturnPacket.size());
    EXPECT_EQ(DATA, rank_->readReturnPacket.front()->busPacketType);
    storeSelected({1});
    expectStored(work, 1, 1);
    EXPECT_TRUE(pim_->allBlocksExited());
}

TEST_F(SelectiveProgressFixture, selection_gates_non_mac_instructions_too)
{
    auto work = makeWorkload({{1, 1}}, 1);
    for (auto& row : work.weights) row[0].fill(2);
    pim_->configureCompletionMask({0, 0});
    prepare(work);
    // ADD x+x has the same reference result as the row of weights all equal to 2.
    pim_->crf.data[0] = PIMCmd(PIMCmdType::ADD, PIMOpdType::GRF_B,
                             PIMOpdType::GRF_A, PIMOpdType::GRF_A).toInt();
    stepSelected(work, 0, {0});
    expectAccumulators(work, {{1, 0}});
    EXPECT_EQ(1u, pim_->getBlockPC(0));
    EXPECT_EQ(0u, pim_->getBlockPC(1));
    storeSelected({0});
    expectStored(work, 0, 1);
    expectUnstored(1);
    stepSelected(work, 0, {1});
    storeSelected({1});
    expectStored(work, 1, 1);
    for (unsigned pb = 0; pb < kBlocks; ++pb)
    {
        EXPECT_EQ(0u, pim_->getExecutedMacs(pb));
        EXPECT_EQ(0u, pim_->getSkippedMacs(pb));
    }
}

TEST_F(SelectiveProgressFixture, enabling_again_preserves_divergent_progress)
{
    const auto work = makeWorkload({{2, 2}}, 2);
    prepare(work);
    stepSelected(work, 0, {0});
    pim_->enableSelectiveProgress();
    EXPECT_EQ(1u, pim_->getBlockPC(0));
    EXPECT_EQ(0u, pim_->getBlockPC(1));
    stepSelected(work, 1, {0});
    storeSelected({0});
    for (unsigned round = 0; round < 2; ++round) stepSelected(work, round, {1});
    storeSelected({1});
    expectStored(work, 0, 2);
    expectStored(work, 1, 2);
}

TEST_F(SelectiveProgressFixture, kernel_restart_resets_private_jump_and_repeat_state)
{
    const auto work = makeWorkload({{8, 8}}, 8);
    for (Program program : {Program::AutoRepeat, Program::JumpRepeat})
    {
        SCOPED_TRACE(program == Program::AutoRepeat ? "auto-repeat" : "jump");
        prepare(work, program);
        stepSelected(work, 0, {0});
        stepSelected(work, 1, {0});
        stepSelected(work, 0, {1});
        prepare(work, program);
        for (unsigned round = 0; round < 8; ++round) stepSelected(work, round, {0, 1});
        storeSelected({0, 1});
        for (unsigned pb = 0; pb < kBlocks; ++pb)
        {
            expectStored(work, pb, 8);
            EXPECT_EQ(8u, pim_->getExecutedMacs(pb));
        }
    }
}
}  // namespace
