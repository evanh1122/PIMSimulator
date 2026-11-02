/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#include "PIMRank.h"

#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "AddressMapping.h"
#include "PIMCmd.h"

using namespace std;
using namespace DRAMSim;

PIMRank::PIMRank(ostream& simLog, Configuration& configuration)
    : chanId(-1),
      rankId(-1),
      dramsimLog(simLog),
      blockStates_(configuration.NUM_PIM_BLOCKS),
      pimOpMode_(false),
      toggleEvenBank_(true),
      toggleOddBank_(true),
      toggleRa13h_(false),
      config(configuration),
      macBudgets_(configuration.NUM_PIM_BLOCKS, 0),
      remainingMacs_(configuration.NUM_PIM_BLOCKS, 0),
      executedMacs_(configuration.NUM_PIM_BLOCKS, 0),
      skippedMacs_(configuration.NUM_PIM_BLOCKS, 0),
      pimBlocks(getConfigParam(UINT, "NUM_PIM_BLOCKS"),
                PIMBlock(PIMConfiguration::getPIMPrecision()))
{
    currentClockCycle = 0;
}

void PIMRank::attachRank(Rank* r)
{
    this->rank = r;
}

void PIMRank::setChanId(int id)
{
    this->chanId = id;
}

void PIMRank::setRankId(int id)
{
    this->rankId = id;
}

int PIMRank::getChanId() const
{
    return this->chanId;
}

int PIMRank::getRankId() const
{
    return this->rankId;
}

void PIMRank::update() {}

void PIMRank::resetMacAccounting()
{
    remainingMacs_ = macBudgets_;
    fill(executedMacs_.begin(), executedMacs_.end(), 0);
    fill(skippedMacs_.begin(), skippedMacs_.end(), 0);
}

void PIMRank::configureCompletionMask(const vector<unsigned>& macCounts)
{
    if (macCounts.size() != pimBlocks.size())
        throw invalid_argument("Completion masking requires one MAC budget per PIM block");

    macBudgets_ = macCounts;
    completionMaskEnabled_ = true;
    resetMacAccounting();
}

void PIMRank::disableCompletionMask()
{
    completionMaskEnabled_ = false;
}

uint64_t PIMRank::getExecutedMacs(unsigned pb) const
{
    return executedMacs_.at(pb);
}

uint64_t PIMRank::getSkippedMacs(unsigned pb) const
{
    return skippedMacs_.at(pb);
}

void PIMRank::enableSelectiveProgress()
{
    if (!selectiveProgressEnabled_)
    {
        // Preserve the current shared position if enabled after execution began.
        fill(blockStates_.begin(), blockStates_.end(), sharedState_);
        selectiveProgressEnabled_ = true;
    }
}

unsigned PIMRank::getBlockPC(unsigned pb) const
{
    const ExecutionState& state = blockStates_.at(pb);
    return selectiveProgressEnabled_ ? state.pc : sharedState_.pc;
}

bool PIMRank::isBlockExited(unsigned pb) const
{
    const ExecutionState& state = blockStates_.at(pb);
    return selectiveProgressEnabled_ ? state.exited : sharedState_.exited;
}

bool PIMRank::allBlocksExited() const
{
    if (!selectiveProgressEnabled_) return sharedState_.exited;
    return all_of(blockStates_.begin(), blockStates_.end(),
                  [](const ExecutionState& state) { return state.exited; });
}

void PIMRank::controlPIM(BusPacket* packet)
{
    uint8_t grf_a_zeroize = packet->data->u8Data_[20];
    if (grf_a_zeroize)
    {
        if (DEBUG_CMD_TRACE)
        {
            PRINTC(RED, OUTLOG_CH_RA("GRF_A_ZEROIZE"));
        }
        BurstType burst_zero;
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            for (int i = 0; i < 8; i++) pimBlocks[pb].grfA[i] = burst_zero;
        }
    }
    uint8_t grf_b_zeroize = packet->data->u8Data_[21];
    if (grf_b_zeroize)
    {
        if (DEBUG_CMD_TRACE)
        {
            PRINTC(RED, OUTLOG_CH_RA("GRF_B_ZEROIZE"));
        }
        BurstType burst_zero;
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            for (int i = 0; i < 8; i++) pimBlocks[pb].grfB[i] = burst_zero;
        }
    }
    pimOpMode_ = packet->data->u8Data_[0] & 1;
    toggleEvenBank_ = !(packet->data->u8Data_[16] & 1);
    toggleOddBank_ = !(packet->data->u8Data_[16] & 2);
    toggleRa13h_ = (packet->data->u8Data_[16] & 4);

    if (pimOpMode_)
    {
        rank->mode_ = dramMode::HAB_PIM;
        sharedState_ = ExecutionState{};
        fill(blockStates_.begin(), blockStates_.end(), ExecutionState{});
        resetMacAccounting();
        PRINTC(RED, OUTLOG_CH_RA("HAB_PIM"));
    }
    else
    {
        rank->mode_ = dramMode::HAB;
        PRINTC(RED, OUTLOG_CH_RA("HAB mode"));
    }
}

bool PIMRank::isToggleCond(BusPacket* packet)
{
    if (pimOpMode_ && !allBlocksExited())
    {
        if (toggleRa13h_)
        {
            if (toggleEvenBank_ && ((packet->bank & 1) == 0))
                return true;
            else if (toggleOddBank_ && ((packet->bank & 1) == 1))
                return true;
            return false;
        }
        else if (!toggleRa13h_ && !isReservedRA(packet->row))
        {
            if (toggleEvenBank_ && ((packet->bank & 1) == 0))
                return true;
            else if (toggleOddBank_ && ((packet->bank & 1) == 1))
                return true;
            return false;
        }
        return false;
    }
    else
    {
        return false;
    }
}

void PIMRank::readHab(BusPacket* packet)
{
    if (isReservedRA(packet->row))  // ignored
    {
        PRINTC(GRAY, OUTLOG_ALL("READ"));
    }
    else
    {
        PRINTC(GRAY, OUTLOG_ALL("BANK_TO_PIM"));
#ifndef NO_STORAGE
        int grf_id = getGrfIdx(packet->column);
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            rank->banks[pb * 2 + packet->bank].read(packet);
            pimBlocks[pb].grfB[grf_id] = *(packet->data);
        }
#endif
    }
}

void PIMRank::writeHab(BusPacket* packet)
{
    if (packet->row == config.PIM_REG_RA)  // WRIO to PIM Broadcasting
    {
        if (packet->column == 0x00)
            controlPIM(packet);
        if ((0x08 <= packet->column && packet->column <= 0x0f) ||
            (0x18 <= packet->column && packet->column <= 0x1f))
        {
            if (DEBUG_CMD_TRACE)
            {
                if (packet->column - 8 < 8)
                    PRINTC(GREEN, OUTLOG_B_GRF_A("BWRITE_GRF_A"));
                else
                    PRINTC(GREEN, OUTLOG_B_GRF_B("BWRITE_GRF_B"));
            }
#ifndef NO_STORAGE
            for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
            {
                if (packet->column - 8 < 8)
                    pimBlocks[pb].grfA[packet->column - 0x8] = *(packet->data);
                else
                    pimBlocks[pb].grfB[packet->column - 0x18] = *(packet->data);
            }
#endif
        }
        else if (0x04 <= packet->column && packet->column <= 0x07)
        {
            if (DEBUG_CMD_TRACE)
                PRINTC(GREEN, OUTLOG_B_CRF("BWRITE_CRF"));
            crf.bst[packet->column - 0x04] = *(packet->data);
        }
        else if (packet->column == 0x1)
        {
            if (DEBUG_CMD_TRACE)
                PRINTC(GREEN, OUTLOG_CH_RA("BWRITE_SRF"));
            for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++) pimBlocks[pb].srf = *(packet->data);
        }
    }
    else if (isReservedRA(packet->row))
    {
        PRINTC(GRAY, OUTLOG_ALL("WRITE"));
    }
    else  // PIM (only GRF) to Bank Move
    {
        PRINTC(GREEN, OUTLOG_ALL("PIM_TO_BANK"));

#ifndef NO_STORAGE
        int grf_id = getGrfIdx(packet->column);
        for (int pb = 0; pb < config.NUM_PIM_BLOCKS; pb++)
        {
            if (packet->bank == 0)
            {
                *(packet->data) = pimBlocks[pb].grfA[grf_id];
                rank->banks[pb * 2].write(packet);  // basically read from bank;
            }
            else if (packet->bank == 1)
            {
                *(packet->data) = pimBlocks[pb].grfB[grf_id];
                rank->banks[pb * 2 + 1].write(packet);  // basically read from bank.
            }
        }
#endif
    }
}

void PIMRank::readOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx,
                      bool is_auto, bool is_mac)
{
    idx = getGrfIdx(idx);

    switch (type)
    {
        case PIMOpdType::A_OUT:
            bst = pimBlocks[pb].aOut;
            return;
        case PIMOpdType::M_OUT:
            bst = pimBlocks[pb].mOut;
            return;
        case PIMOpdType::EVEN_BANK:
            if (packet->bank % 2 != 0)
                PRINT("Warning, CRF bank coding and bank id from packet are inconsistent");
            rank->banks[pb * 2].read(packet);  // basically read from bank.
            bst = *(packet->data);
            return;
        case PIMOpdType::ODD_BANK:
            if (packet->bank % 2 == 0)
                PRINT("Warning, CRF bank coding and bank id from packet are inconsistent");
            rank->banks[pb * 2 + 1].read(packet);  // basically read from bank.
            bst = *(packet->data);
            return;
        case PIMOpdType::GRF_A:
            bst = pimBlocks[pb].grfA[(is_auto) ? getGrfIdx(packet->column) : idx];
            return;
        case PIMOpdType::GRF_B:
            if (is_auto)
                bst = pimBlocks[pb].grfB[(is_mac) ? getGrfIdxHigh(packet->row, packet->column)
                                                  : getGrfIdx(packet->column)];
            else
                bst = pimBlocks[pb].grfB[idx];
            return;
        case PIMOpdType::SRF_M:
            bst.set(pimBlocks[pb].srf.fp16Data_[idx]);
            return;
        case PIMOpdType::SRF_A:
            bst.set(pimBlocks[pb].srf.fp16Data_[idx + 8]);
            return;
    }
}

void PIMRank::writeOpd(int pb, BurstType& bst, PIMOpdType type, BusPacket* packet, int idx,
                       bool is_auto, bool is_mac)
{
    idx = getGrfIdx(idx);

    switch (type)
    {
        case PIMOpdType::A_OUT:
            pimBlocks[pb].aOut = bst;
            return;
        case PIMOpdType::M_OUT:
            pimBlocks[pb].mOut = bst;
            return;
        case PIMOpdType::EVEN_BANK:
            if (packet->bank % 2 != 0)
            {
                PRINT("CRF bank coding and bank id from packet are inconsistent");
            }
            *(packet->data) = bst;
            rank->banks[pb * 2].write(packet);  // basically read from bank.
            return;
        case PIMOpdType::ODD_BANK:
            if (packet->bank % 2 == 0)
            {
                PRINT("CRF bank coding and bank id from packet are inconsistent");
                exit(-1);
            }
            *(packet->data) = bst;
            rank->banks[pb * 2 + 1].write(packet);  // basically read from bank.
            return;
        case PIMOpdType::GRF_A:
            pimBlocks[pb].grfA[(is_auto) ? getGrfIdx(packet->column) : idx] = bst;
            return;
        case PIMOpdType::GRF_B:
            if (is_auto)
                pimBlocks[pb].grfB[(is_mac) ? getGrfIdxHigh(packet->row, packet->column)
                                            : getGrfIdx(packet->column)] = bst;
            else
                pimBlocks[pb].grfB[idx] = bst;
            return;
        case PIMOpdType::SRF_M:
            pimBlocks[pb].srf = bst;
            return;
        case PIMOpdType::SRF_A:
            pimBlocks[pb].srf = bst;
            return;
    }
}

void PIMRank::doPIM(BusPacket* packet)
{
    vector<unsigned> blocks(pimBlocks.size());
    iota(blocks.begin(), blocks.end(), 0);
    if (selectiveProgressEnabled_)
        doPIMSelected(packet, blocks);
    else
        executePIM(packet, sharedState_, blocks);
}

void PIMRank::doPIMSelected(BusPacket* packet, const vector<unsigned>& blocks)
{
    if (!selectiveProgressEnabled_ || !pimOpMode_)
        throw logic_error("Selective progress requires enabling it and entering PIM mode");
    if (packet == nullptr || (packet->busPacketType != READ && packet->busPacketType != WRITE))
        throw invalid_argument("Selective progress requires a READ or WRITE packet");

    // Validate the entire selection before any block or packet data changes.
    vector<bool> seen(pimBlocks.size(), false);
    for (unsigned pb : blocks)
    {
        if (pb >= pimBlocks.size()) throw out_of_range("Selected PIM block is out of range");
        if (seen[pb]) throw invalid_argument("Selected PIM blocks must be unique");
        seen[pb] = true;
    }

    // Operand reads/stores use packet->data as scratch. Give each block the
    // original request data so one block cannot feed its scratch into another.
    const BurstType requestData = packet->data ? *packet->data : BurstType{};
    for (unsigned pb : blocks)
    {
        if (blockStates_[pb].exited) continue;
        BurstType localData = requestData;
        BusPacket localPacket = *packet;
        localPacket.data = &localData;
        executePIM(&localPacket, blockStates_[pb], vector<unsigned>{pb});
        if (packet->data) *packet->data = localData;
    }
}

void PIMRank::executePIM(BusPacket* packet, ExecutionState& state, const vector<unsigned>& blocks)
{
    if (state.exited) return;
    packet->row = masked2accessibleRA(packet->row);
    const auto fetch = [this](int pc)
    {
        if (pc < 0 || static_cast<unsigned>(pc) >= sizeof(crf.data) / sizeof(crf.data[0]))
            throw out_of_range("PIM program counter is outside the CRF");
        PIMCmd command;
        command.fromInt(crf.data[pc]);
        return command;
    };

    PIMCmd cCmd;
    do
    {
        cCmd = fetch(state.pc);
        if (DEBUG_CMD_TRACE)
        {
            PRINTC(CYAN, string((packet->busPacketType == READ) ? "READ ch" : "WRITE ch")
                             << getChanId() << " ra" << getRankId() << " bg"
                             << config.addrMapping.bankgroupId(packet->bank) << " b" << packet->bank
                             << " r" << packet->row << " c" << packet->column << "|| [" << state.pc
                             << "] " << cCmd.toStr() << " pb["
                             << (blocks.size() == 1 ? to_string(blocks.front()) : "all")
                             << "] @ " << currentClockCycle);
        }

        if (cCmd.type_ == PIMCmdType::EXIT)
        {
            state.exited = true;
            break;
        }
        else if (cCmd.type_ == PIMCmdType::JUMP)
        {
            if (state.lastJumpIdx != state.pc)
            {
                if (cCmd.loopCounter_ > 0)
                {
                    state.lastJumpIdx = state.pc;
                    state.remainingJumps = cCmd.loopCounter_;
                }
            }
            if (state.remainingJumps > 0)
            {
                state.pc -= cCmd.loopOffset_;
                state.remainingJumps--;
            }
        }
        else
        {
            if (cCmd.type_ == PIMCmdType::FILL || cCmd.isAuto_)
            {
                if (state.lastRepeatIdx != state.pc)
                {
                    state.lastRepeatIdx = state.pc;
                    state.remainingRepeats = 8 - 1;
                }

                if (state.remainingRepeats > 0)
                {
                    state.pc -= 1;
                    state.remainingRepeats--;
                }
                else
                    state.lastRepeatIdx = -1;
            }
            else if (cCmd.type_ == PIMCmdType::NOP)
            {
                if (state.lastRepeatIdx != state.pc)
                {
                    state.lastRepeatIdx = state.pc;
                    state.remainingRepeats = cCmd.loopCounter_;
                }

                if (state.remainingRepeats > 0)
                {
                    state.pc -= 1;
                    state.remainingRepeats--;
                }
                else
                    state.lastRepeatIdx = -1;
            }

            for (unsigned pb : blocks) executeBlockInstruction(packet, cCmd, pb);
        }
        state.pc++;
        // EXIT is observed before the next trigger, after any pending store.
        if (fetch(state.pc).type_ == PIMCmdType::EXIT) state.exited = true;
    } while (cCmd.type_ == PIMCmdType::JUMP && !state.exited);
}

void PIMRank::executeBlockInstruction(BusPacket* packet, const PIMCmd& command, unsigned pb)
{
    const bool isMac = command.type_ == PIMCmdType::MAC;
    if (isMac && completionMaskEnabled_ && remainingMacs_[pb] == 0)
    {
        ++skippedMacs_[pb];
        return;
    }

    doPIMBlock(packet, command, pb);

    if (isMac)
    {
        ++executedMacs_[pb];
        if (completionMaskEnabled_) --remainingMacs_[pb];
    }

    if (DEBUG_PIM_BLOCK && pb == 0)
    {
        PRINT("[BANK_R]" << packet->data->fp16ToStr());
        PRINT("[CMD]" << bitset<32>(command.toInt()) << "(" << command.toStr() << ")");
        PRINT(pimBlocks[pb].print());
        PRINT("----------");
    }
}

void PIMRank::doPIMBlock(BusPacket* packet, PIMCmd cCmd, int pimblock_id)
{
    if (cCmd.type_ == PIMCmdType::FILL || cCmd.type_ == PIMCmdType::MOV)
    {
        BurstType bst;
        bool is_auto = (cCmd.type_ == PIMCmdType::FILL) ? true : false;

        readOpd(pimblock_id, bst, cCmd.src0_, packet, cCmd.src0Idx_, is_auto, false);
        if (cCmd.isRelu_)
        {
            for (int i = 0; i < 16; i++)
                bst.u16Data_[i] = (bst.u16Data_[i] & (1 << 15)) ? 0 : bst.u16Data_[i];
        }
        writeOpd(pimblock_id, bst, cCmd.dst_, packet, cCmd.dstIdx_, is_auto, false);
    }
    else if (cCmd.type_ == PIMCmdType::ADD || cCmd.type_ == PIMCmdType::MUL)
    {
        BurstType dstBst;
        BurstType src0Bst;
        BurstType src1Bst;
        readOpd(pimblock_id, src0Bst, cCmd.src0_, packet, cCmd.src0Idx_, cCmd.isAuto_, false);
        readOpd(pimblock_id, src1Bst, cCmd.src1_, packet, cCmd.src1Idx_, cCmd.isAuto_, false);

        if (cCmd.type_ == PIMCmdType::ADD)
            // dstBst = src0Bst + src1Bst;
            pimBlocks[pimblock_id].add(dstBst, src0Bst, src1Bst);
        else if (cCmd.type_ == PIMCmdType::MUL)
            // dstBst = src0Bst * src1Bst;
            pimBlocks[pimblock_id].mul(dstBst, src0Bst, src1Bst);

        writeOpd(pimblock_id, dstBst, cCmd.dst_, packet, cCmd.dstIdx_, cCmd.isAuto_, false);
    }
    else if (cCmd.type_ == PIMCmdType::MAC || cCmd.type_ == PIMCmdType::MAD)
    {
        BurstType dstBst;
        BurstType src0Bst;
        BurstType src1Bst;
        bool is_mac = (cCmd.type_ == PIMCmdType::MAC) ? true : false;

        readOpd(pimblock_id, src0Bst, cCmd.src0_, packet, cCmd.src0Idx_, cCmd.isAuto_, is_mac);
        readOpd(pimblock_id, src1Bst, cCmd.src1_, packet, cCmd.src1Idx_, cCmd.isAuto_, is_mac);
        if (is_mac)
        {
            readOpd(pimblock_id, dstBst, cCmd.dst_, packet, cCmd.dstIdx_, cCmd.isAuto_, is_mac);
            // dstBst = src0Bst * src1Bst + dstBst;
            pimBlocks[pimblock_id].mac(dstBst, src0Bst, src1Bst);
        }
        else
        {
            BurstType src2Bst;
            readOpd(pimblock_id, src2Bst, cCmd.src2_, packet, cCmd.src1Idx_, cCmd.isAuto_, is_mac);
            // dstBst = src0Bst * src1Bst + src2Bst;
            pimBlocks[pimblock_id].mad(dstBst, src0Bst, src1Bst, src2Bst);
        }

        writeOpd(pimblock_id, dstBst, cCmd.dst_, packet, cCmd.dstIdx_, cCmd.isAuto_, is_mac);
    }
    else if (cCmd.type_ == PIMCmdType::NOP && packet->busPacketType == WRITE)
    {
        int grf_id = getGrfIdx(packet->column);
        if (packet->bank == 0)
        {
            *(packet->data) = pimBlocks[pimblock_id].grfA[grf_id];
            rank->banks[pimblock_id * 2].write(packet);  // basically read from bank;
        }
        else if (packet->bank == 1)
        {
            *(packet->data) = pimBlocks[pimblock_id].grfB[grf_id];
            rank->banks[pimblock_id * 2 + 1].write(packet);  // basically read from bank.
        }
    }
}
