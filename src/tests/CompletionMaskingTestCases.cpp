// Functional tests for per-block MAC completion. These drive the real PIMRank
// interpreter and bank storage directly; they do not measure DRAM timing.
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ConfigurationDB.h"
#include "PIMRank.h"
#include "Rank.h"
#include "gtest/gtest.h"

namespace
{
using namespace DRAMSim;

constexpr unsigned kBlocks = 2;
constexpr unsigned kLanes = 16;
using Vector = std::array<float, kLanes>;

struct Workload
{
    // Each block owns one matrix row; each round consumes 16 adjacent columns.
    std::vector<Vector> input;
    std::array<std::vector<Vector>, kBlocks> weights;
    std::array<unsigned, kBlocks> validMacs;
};

Workload makeWorkload(std::array<unsigned, kBlocks> validMacs, unsigned rounds,
                      bool poisonPadding = false)
{
    Workload work;
    work.validMacs = validMacs;
    work.input.resize(rounds);
    for (auto& row : work.weights) row.resize(rounds);
    for (unsigned round = 0; round < rounds; ++round)
    {
        for (unsigned lane = 0; lane < kLanes; ++lane)
        {
            work.input[round][lane] = 1 + (round + lane) % 2;
            for (unsigned pb = 0; pb < kBlocks; ++pb)
            {
                work.weights[pb][round][lane] = round < validMacs[pb]
                    ? 1 + (pb + round + lane) % 3
                    : (poisonPadding ? 7 : 0);
            }
        }
    }
    return work;
}

// Host-side discovery in the same order as the compute triggers. Keep internal
// zero chunks in place; only the suffix after the last useful chunk is masked.
// Check FP16 values, matching the conversion performed when weights are loaded.
std::array<unsigned, kBlocks> discoverMacBudgets(const Workload& work)
{
    std::array<unsigned, kBlocks> budgets{};
    for (unsigned pb = 0; pb < kBlocks; ++pb)
        for (unsigned chunk = 0; chunk < work.weights[pb].size(); ++chunk)
            if (std::any_of(work.weights[pb][chunk].begin(), work.weights[pb][chunk].end(),
                            [](float weight) { return convertH2F(convertF2H(weight)) != 0.0f; }))
                budgets[pb] = chunk + 1;
    return budgets;
}

// Independent CPU reference: deliberately does not call the simulator's MAC.
// Small integer operands make every intermediate exactly representable in FP16.
std::array<double, kLanes> reference(const Workload& work, unsigned pb, unsigned macs)
{
    std::array<double, kLanes> result{};
    for (unsigned round = 0; round < macs; ++round)
        for (unsigned lane = 0; lane < kLanes; ++lane)
            result[lane] += double(work.weights[pb][round][lane]) * work.input[round][lane];
    return result;
}

enum class Program { Unrolled, AutoRepeat, JumpRepeat };

class PIMControlFixture : public testing::Test
{
  protected:
    void SetUp() override
    {
#ifdef NO_STORAGE
        GTEST_SKIP() << "These correctness tests require storage; rebuild with NO_STORAGE=0.";
#endif
        // A small functional configuration: two blocks, each beside two banks.
        auto& db = ConfigurationDB::getDB();
        db.initialize();
        db.updatefromFile("ini/HBM2_samsung_2M_16B_x64.ini");
        db.updatefromFile("system_hbm_1ch.ini");
        setDevConfigParam(UINT, "NUM_BANKS", 4);
        setDevConfigParam(UINT, "NUM_BANK_GROUPS", 1);
        setDevConfigParam(UINT, "NUM_PIM_BLOCKS", kBlocks);
        setDevConfigParam(UINT, "NUM_RANKS", 1);
        setSysConfigParam(BOOL, "DEBUG_CMD_TRACE", false);
        mapping_.reset(new AddrMapping());
        config_.reset(new Configuration(*mapping_));
        rank_.reset(new Rank(log_, *config_));
        rank_->setChanId(0);
        rank_->setRankId(0);
        pim_ = rank_->pimRank.get();
        pim_->setChanId(0);
        pim_->setRankId(0);
    }

    void configureMask(const Workload& work)
    {
        pim_->configureCompletionMask(std::vector<unsigned>(
            work.validMacs.begin(), work.validMacs.end()));
    }

    void prepare(const Workload& work, Program program = Program::Unrolled)
    {
        const unsigned rounds = work.input.size();
        ASSERT_GE(rounds, 1u);
        ASSERT_LE(rounds, 8u);
        for (unsigned pb = 0; pb < kBlocks; ++pb)
        {
            ASSERT_LE(work.validMacs[pb], rounds);
            for (unsigned round = 0; round < rounds; ++round)
            {
                BurstType data;
                for (unsigned lane = 0; lane < kLanes; ++lane)
                    data.fp16Data_[lane] = convertF2H(work.weights[pb][round][lane]);
                BusPacket packet(WRITE, 0, round, 0, 0, 2 * pb, &data, log_);
                rank_->banks[2 * pb].write(&packet);
            }
            // A sentinel proves that the store happens, even for an empty block.
            BurstType sentinel;
            sentinel.set(convertF2H(-123));
            BusPacket packet(WRITE, 0, 0, 2, 0, 2 * pb + 1, &sentinel, log_);
            rank_->banks[2 * pb + 1].write(&packet);
        }

        std::vector<PIMCmd> commands;
        PIMCmd mac(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                   PIMOpdType::EVEN_BANK, program == Program::AutoRepeat ? 1 : 0);
        if (program == Program::Unrolled)
            commands.assign(rounds, mac);
        else if (program == Program::AutoRepeat)
        {
            ASSERT_EQ(8u, rounds);  // Auto mode repeats across eight triggers.
            commands.push_back(mac);
        }
        else
        {
            commands.push_back(mac);
            commands.emplace_back(PIMCmdType::JUMP, rounds - 1, 2);
        }
        // The write-triggered NOP stores GRF_B[0] to each block's odd bank.
        commands.emplace_back(PIMCmdType::NOP, 0);
        commands.emplace_back(PIMCmdType::EXIT, 0);
        for (unsigned burst = 0; burst < 4; ++burst)
        {
            BurstType data;
            for (unsigned slot = 0; slot < 8; ++slot)
            {
                unsigned index = burst * 8 + slot;
                data.u32Data_[slot] = index < commands.size()
                    ? commands[index].toInt() : PIMCmd(PIMCmdType::EXIT, 0).toInt();
            }
            BusPacket packet(WRITE, 0, 4 + burst, config_->PIM_REG_RA, 0, 0, &data, log_);
            pim_->writeHab(&packet);
        }

        BurstType control;
        control.u8Data_[0] = 1;
        control.u8Data_[21] = 1;  // Clear accumulators at this new kernel start.
        BusPacket packet(WRITE, 0, 0, config_->PIM_REG_RA, 0, 0, &control, log_);
        pim_->controlPIM(&packet);
        program_ = program;
    }

    void step(const Workload& work, unsigned round)
    {
        BurstType input;
        for (unsigned lane = 0; lane < kLanes; ++lane)
            input.fp16Data_[lane] = convertF2H(work.input[round][lane]);
        const unsigned reg = program_ == Program::AutoRepeat ? round : 0;
        BusPacket load(WRITE, 0, 0x8 + reg, config_->PIM_REG_RA, 0, 0, &input, log_);
        pim_->writeHab(&load);  // Broadcast the matching vector chunk.

        BurstType scratch;
        BusPacket trigger(READ, 0, round, 0, 0, 0, &scratch, log_);
        ASSERT_TRUE(pim_->isToggleCond(&trigger)) << "Shared program ended before its final store";
        pim_->doPIM(&trigger);
    }

    void expectAccumulators(const Workload& work, std::array<unsigned, kBlocks> counts)
    {
        for (unsigned pb = 0; pb < kBlocks; ++pb)
        {
            const auto expected = reference(work, pb, counts[pb]);
            for (unsigned lane = 0; lane < kLanes; ++lane)
                EXPECT_DOUBLE_EQ(expected[lane], convertH2F(pim_->pimBlocks[pb].grfB[0].fp16Data_[lane]))
                    << "block=" << pb << " lane=" << lane;
        }
    }

    void storeAndCheck(const Workload& work, std::array<unsigned, kBlocks> counts)
    {
        BurstType scratch;
        BusPacket store(WRITE, 0, 0, 2, 0, 1, &scratch, log_);
        ASSERT_TRUE(pim_->isToggleCond(&store));
        pim_->doPIM(&store);
        EXPECT_FALSE(pim_->isToggleCond(&store)) << "Expected EXIT after the result store";
        for (unsigned pb = 0; pb < kBlocks; ++pb)
        {
            BurstType output;
            output.set(convertF2H(-456));
            BusPacket read(READ, 0, 0, 2, 0, 2 * pb + 1, &output, log_);
            rank_->banks[2 * pb + 1].read(&read);
            const auto expected = reference(work, pb, counts[pb]);
            double actualDotProduct = 0, expectedDotProduct = 0;
            for (unsigned lane = 0; lane < kLanes; ++lane)
            {
                EXPECT_DOUBLE_EQ(expected[lane], convertH2F(output.fp16Data_[lane]))
                    << "stored block=" << pb << " lane=" << lane;
                actualDotProduct += convertH2F(output.fp16Data_[lane]);
                expectedDotProduct += expected[lane];
            }
            EXPECT_DOUBLE_EQ(expectedDotProduct, actualDotProduct);
        }
    }

    void run(const Workload& work, bool masked, Program program = Program::Unrolled)
    {
        prepare(work, program);
        for (unsigned round = 0; round < work.input.size(); ++round)
        {
            step(work, round);
            std::array<unsigned, kBlocks> counts;
            for (unsigned pb = 0; pb < kBlocks; ++pb)
                counts[pb] = masked ? std::min(round + 1, work.validMacs[pb]) : round + 1;
            expectAccumulators(work, counts);
        }
        const unsigned rounds = work.input.size();
        storeAndCheck(work, masked ? work.validMacs : std::array<unsigned, kBlocks>{{rounds, rounds}});
        if (masked)
        {
            for (unsigned pb = 0; pb < kBlocks; ++pb)
            {
                EXPECT_EQ(work.validMacs[pb], pim_->getExecutedMacs(pb));
                EXPECT_EQ(rounds - work.validMacs[pb], pim_->getSkippedMacs(pb));
            }
        }
        else
        {
            for (unsigned pb = 0; pb < kBlocks; ++pb)
            {
                EXPECT_EQ(rounds, pim_->getExecutedMacs(pb));
                EXPECT_EQ(0u, pim_->getSkippedMacs(pb));
            }
        }
    }

    std::ostringstream log_;
    std::unique_ptr<AddrMapping> mapping_;
    std::unique_ptr<Configuration> config_;
    std::unique_ptr<Rank> rank_;
    PIMRank* pim_ = nullptr;
    Program program_ = Program::Unrolled;
};

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
