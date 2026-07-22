# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)
        tl.store(num_rejected_tokens_gpu_ptr + offsets, num_rejected, mask=mask)


# 输入全拍平(一维的)、计算靠指针寻址
@triton.jit
def copy_and_expand_dflash_inputs_kernel_single_grid(
    # Inputs 输入张量指针
    next_token_ids_ptr,  # [num_reqs] 每个请求基准draft token ID（第一根query使用）
    target_positions_ptr,  # [num_context] 所有请求上下文token对应的全局位置pos
    # Outputs 输出张量指针
    out_input_ids_ptr,  # [num_query_total] 拼接后完整输入ID：1个真实draft + N个占位draft token
    out_context_positions_ptr,  # [num_context] 拷贝后的上下文pos，原样输出
    out_query_positions_ptr,  # [num_query_total] 每一条query对应的全局解码位置pos
    out_context_slot_mapping_ptr,  # [num_context] 上下文token对应的KV cache物理slot索引
    out_query_slot_mapping_ptr,  # [num_query_total] query候选token对应的KV cache物理slot索引
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] 记录每条spec token在query输出数组中的下标
    # Block table KV分页块表相关
    block_table_ptr,  # [max_reqs, max_blocks] 全局块表：req_id -> block_id映射
    block_table_stride,  # block_table第0维步长（单个请求占用block数量）
    # Metadata 批次元数据
    query_start_loc_ptr,  # [num_reqs + 1] 前缀和数组：每个请求上下文token起始/结束下标
    num_rejected_tokens_ptr,  # [num_reqs] 本轮被拒绝的spec token数量，无padding时传0/null
    # Scalars 标量超参
    parallel_drafting_token_id,  # tl.int32 占位draft token id（除第一个真实draft外其余query填充此值）
    block_size,  # tl.int32 KV Cache单块容纳token数
    num_query_per_req,  # tl.int32 单请求并行生成的候选query总数 = num_speculative_tokens + 1
    num_speculative_tokens,  # tl.int32 开启推测解码的并行draft token数量
    total_input_tokens,  # tl.int32 全局所有上下文token总长度
    batch_size,  # tl.int32 当前批次请求总数
    HAS_NUM_REJECTED: tl.constexpr = False,  # 编译期常量：是否存在被拒绝token，控制上下文有效截断
):
    # 遍历批次内每一条推理请求
    for req_idx in range(0, batch_size):
        # 读取当前请求上下文token在target_positions中的起始下标
        ctx_start = tl.load(query_start_loc_ptr + req_idx)
        # 读取当前请求上下文token在target_positions中的结束下标（开区间）
        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
        # 计算当前请求上下文token总个数
        num_ctx = ctx_end - ctx_start

        # 遍历当前请求所有上下文token，拷贝pos并计算KV slot映射
        for j in range(0, num_ctx):
            # 当前上下文token在全局上下文数组中的线性下标
            ctx_pos_idx = ctx_start + j
            # 加载该token全局解码位置pos
            pos = tl.load(target_positions_ptr + ctx_pos_idx)
            # 将原始pos拷贝输出到上下文pos数组
            tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

            # 根据pos计算该token落在第几块KV Block
            block_num = pos // block_size
            # 从块表读取该pos对应的物理Block ID
            block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_num).to(tl.int64)
            # 计算该token在KV Cache中的物理slot编号
            slot = block_id * block_size + (pos % block_size)
            # 保存上下文token的slot映射
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

        # 判断是否存在被拒绝token，截断有效上下文长度
        if HAS_NUM_REJECTED:
            # 加载当前请求本轮废弃的spec token数量
            num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
            # 有效上下文终点 = 原终点 - 被丢弃token数，跳过失效历史token
            valid_ctx_end = ctx_end - num_rejected
        else:
            # 无拒绝token，全部上下文有效
            valid_ctx_end = ctx_end

        # 获取当前请求有效上下文最后一个token的全局pos
        last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)

        # 遍历当前请求所有并行query候选（1真实draft + N占位spec token）
        for q_idx in range(0, num_query_per_req):
            # 当前候选query对应的全局解码pos：最后上下文pos向后顺延
            query_pos = last_pos + 1 + q_idx
            # 当前query在全局query输出数组中的线性下标
            query_out_idx = req_idx * num_query_per_req + q_idx

            # 存储该query对应的全局解码位置
            tl.store(out_query_positions_ptr + query_out_idx, query_pos)

            # 计算该query pos所属KV块编号
            block_num_q = query_pos // block_size
            # 读取该块对应的物理Block ID
            block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q).to(tl.int64)
            # 计算该候选token对应的KV物理slot
            slot_q = block_id_q * block_size + (query_pos % block_size)
            # 保存query侧slot映射，用于注意力读取KV Cache
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

            # 分支：第一个query使用真实draft token，其余填充占位token
            if q_idx == 0:
                # 加载该请求真实预生成draft token id
                bonus_token = tl.load(next_token_ids_ptr + req_idx)
                # 第一个query填入真实draft token
                tl.store(out_input_ids_ptr + query_out_idx, bonus_token)
            else:
                # 非首条query填充占位draft token
                tl.store(out_input_ids_ptr + query_out_idx, parallel_drafting_token_id)

                # 计算该spec token在out_token_indices中的存储下标
                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                # 记录这条spec token对应的query输出下标，后续校验阶段定位
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)