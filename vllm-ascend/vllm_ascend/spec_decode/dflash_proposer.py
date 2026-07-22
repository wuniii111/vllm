from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_inputs_kernel_single_grid
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


# 继承昇腾平台 Eagle 推测解码基类，实现 DFlash 优化版推测解码提案器
# monkey_patch
class AscendDflashProposer(AscendEagleProposer):
    # 构造函数：初始化 DFlash 推测解码所需缓冲区、配置与张量
    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置对象，包含批大小、序列长度、模型参数等
        device: torch.device,     # 计算设备（昇腾 NPU 设备）
        runner=None,               # vLLM 运行时管理器，负责调度、KV Cache、批处理等
    ):
        # 调用父类 AscendEagleProposer 的构造方法，完成基础初始化
        super().__init__(
            vllm_config,
            device,
            runner=runner,
        )

        # 单轮查询侧最大 token 数 = 最大批大小 * (1 + 推测解码每轮生成 token 数)
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # 全局最大位置编码长度 = 上下文最大 token 数 + 查询侧最大 token 数
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # 预分配一块 固定 的连续内存空间
        # 上下文侧 slot 映射缓冲区：映射 token 到 KV Cache 物理 slot，int32 类型，NPU 上分配
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        # 查询侧 slot 映射缓冲区：推测解码查询分支使用的 slot 映射
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        # 上下文位置编码缓冲区：存储上下文 token 对应的 position id
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device=device,
        )

        # 查询分支位置编码张量：存储推测 token 对应的 position id
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )

        # 预生成连续索引数组：用于快速构造 query_start_loc 等分段偏移，避免运行时 arange
        self.arange_dflash = torch.arange(self.max_positions + 1, device=device, dtype=torch.int32)

        # DFlash 上下文隐态缓冲区：存放主模型上下文的 hidden states，用于交叉注意力 K/V
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
        )

        # 并行解码 draft 分支隐态张量，初始置空，运行时动态赋值
        self.parallel_drafting_hidden_state_tensor = None

    # 首轮输入预处理：组装 DFlash 交叉注意力所需输入、更新注意力元数据、调用自定义核函数
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,          # 主模型上下文 token id 序列
        next_token_ids: torch.Tensor,           # 待生成的下一批推测 token id
        target_positions: torch.Tensor,         # 主模型上下文位置编码
        target_hidden_states: torch.Tensor,     # 主模型上下文输出隐态（作为交叉注意力 K/V）
        token_indices_to_sample: torch.Tensor | None,  # 需要采样的 token 索引列表
        cad: CommonAttentionMetadata,            # 通用注意力元数据（KV Cache、序列长度、起止位置等）
        num_rejected_tokens_gpu: torch.Tensor | None,  # 本轮被拒绝的推测 token 数量（GPU 张量）
        req_scheduled_tokens=None,              # 调度请求对应的 token 计数
        long_seq_metadata=None,                 # 长序列特殊处理元数据
        num_prefill_reqs=0,                     # Prefill 阶段请求数
        num_decode_reqs=0,                      # Decode 阶段请求数
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # DFlash 交叉注意力说明：上下文 K/V 来自主模型隐态，Query 来自推测分支嵌入
        batch_size = cad.num_reqs  # 当前批内请求总数
        num_context = target_token_ids.shape[0]  # 上下文侧总 token 数量
        num_query_per_req = 1 + self.num_speculative_tokens  # 每个请求对应的查询侧 token 数
        num_query_total = batch_size * num_query_per_req      # 查询侧全局总 token 数

        # 记录当前上下文 token 数量，供后续流程使用
        self._dflash_num_context = num_context
        # 将主模型上下文隐态拷贝到 DFlash 专用隐态缓冲区
        self._dflash_hidden_states[:num_context] = target_hidden_states

        # 重新初始化采样索引张量：维度 = 批大小 * 单请求推测 token 数
        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # 判断是否存在被拒绝的推测 token 计数张量
        has_num_rejected = num_rejected_tokens_gpu is not None

        # 调用昇腾自定义 Kernel：单网格核函数，完成输入拷贝、扩展、slot/位置映射等批量计算
        # 把输入的 next_token_ids 拷贝并扩展到 self.input_ids 中
        # 自动计算好每个请求的 positions（比如请求一对应位置 [10, 11, 12, 13]，请求二对应 [25, 26, 27, 28])
        # 把它们在 KV Cache 里的物理柜子号（slot_mapping）填好
        copy_and_expand_dflash_inputs_kernel_single_grid[1,](
            # 输入张量
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # 输出张量：模型输入 id、上下文位置、查询位置、两类 slot 映射、采样索引
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # KV Cache 块表 & 块表步长
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # 注意力分段起始位置
            query_start_loc_ptr=cad.query_start_loc,
            # 被拒绝 token 数（无则传 0）
            num_rejected_tokens_ptr=(num_rejected_tokens_gpu if has_num_rejected else 0),
            # 常量超参
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.kernel_block_size,        # KV Cache 单个块大小
            num_query_per_req=num_query_per_req,      # 单请求查询 token 数
            num_speculative_tokens=self.num_speculative_tokens,  # 每轮推测生成 token 数
            total_input_tokens=num_context,           # 上下文总 token 数
            batch_size=batch_size,                     # 当前批大小
            HAS_NUM_REJECTED=has_num_rejected,         # 标记是否存在拒绝 token
        )

        # 截取有效查询侧 slot 映射
        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        # 重新构造查询分段起始位置：预生成 arange 数组 * 单请求查询 token 数
        new_query_start_loc = self.arange_dflash[: batch_size + 1] * num_query_per_req

        # 原始有效序列长度
        effective_seq_lens = cad.seq_lens
        # 若有拒绝 token，序列长度扣除被拒绝部分
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        # 更新注意力元数据：查询分段起始位置
        cad.query_start_loc = new_query_start_loc
        # 序列长度 = 原有效长度 + 本轮查询新增 token 数
        cad.seq_lens = effective_seq_lens + num_query_per_req
        # CPU 侧查询分段起始位置，同步更新
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * num_query_per_req
        ).to(torch.int32)

        # 若元数据存在 Q 侧实际序列长度，统一赋值为单请求查询 token 数
        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        # 若存在单请求解码 token 计数，进行赋值
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        # 注意力层实际参与计算的 query token 总数
        cad.num_actual_tokens = num_query_total
        # 单请求最大查询长度
        cad.max_query_len = num_query_per_req
        # 全局最大序列长度累加
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        # 绑定查询侧 slot 映射到注意力元数据
        cad.slot_mapping = query_slot_mapping
        # DFlash 交叉注意力不使用因果掩码（非自回归因果注意力）
        cad.causal = False
        # 清空注意力掩码，由核/硬件原生控制
        cad.attn_mask = None
        # 标记注意力状态：分块 Prefill 模式（昇腾 ChunkedPrefill）
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        # 返回：查询总 token 数、采样索引、更新后的注意力元数据、附加元数据(空)
        return num_query_total, token_indices_to_sample, cad, None

    # 推理模式下的空跑/预热函数：用于 CUDAGraph/ACL 图捕获、性能预热、Profile 采样
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,                          # 本轮空跑总 token 数
        num_reqs: int = 0,                        # 本轮批内请求数
        num_tokens_across_dp: torch.Tensor | None = None,  # 数据并行各卡 token 分布
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,  # 昇腾图执行模式
        batch_descriptor=None,                    # 批描述符，记录批属性
        dummy_compute_logits=lambda hidden_states: None,  # 空跑时 logits 计算占位函数
        is_profile=False,                         # 是否为性能采样/Profile 场景
        **kwargs,
    ) -> None:
        # 限制查询 token 数不超过硬件/配置允许的最大值
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        # 跨数据并行同步元数据，获取实际输入 token 数与各卡分布
        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        # 未开启 CUDA/ACL 图则强制关闭图模式
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        # 每个请求对应的查询侧 token 数量
        num_query_per_req = 1 + self.num_speculative_tokens
        # 全局查询侧总 token 数
        num_query_total = num_reqs * num_query_per_req

        # 截取有效上下文位置编码缓冲区
        context_positions = self._context_positions_buffer[:num_input_tokens]
        # 截取有效上下文隐态缓冲区
        context_states = self.hidden_states[:num_input_tokens]

        # 多步注意力元数据列表，用于图捕获场景
        multi_steps_attn_metadata = []
        # 全图捕获模式 且 存在注意力分组时，构造图专用注意力元数据
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            # 获取注意力元数据构造器
            builder = self.draft_attn_groups[0].get_metadata_builder()
            # 初始化昇腾通用注意力元数据
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.arange_dflash[: num_reqs + 1] * num_query_per_req,
                query_start_loc_cpu=torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * num_query_per_req,
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                seq_lens=self.runner.seq_lens[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_query_tokens,
                max_query_len=num_query_per_req,
                max_seq_len=0,
                slot_mapping=self._slot_mapping_buffer[:num_query_total],
                attn_state=AscendAttentionState.ChunkedPrefill,
                causal=False,
                block_table_tensor=self.runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor()[
                    :num_reqs
                ],
            )

            # 为图捕获构建专用注意力元数据
            attn_metadata_dflash = builder.build_for_graph_capture(
                common_attn_metadata,
                AscendAttentionState.ChunkedPrefill,
            )

            # 清空注意力掩码，统一状态
            attn_metadata_dflash.attn_mask = None
            attn_metadata_dflash.attn_state = AscendAttentionState.ChunkedPrefill

            # 按模型层名绑定同一套注意力元数据
            per_layer_attn_metadata = dict()
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata_dflash
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        # 进入昇腾前向上下文管理器，绑定注意力元数据、图模式、并行信息等
        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            # Profile 性能采样场景：预计算并缓存上下文 KV，再执行一次模型前向
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            # 普通空跑/图预热场景：执行 DFlash 可运行逻辑
            else:
                # 记录当前上下文 token 数
                self._dflash_num_context = num_input_tokens
                # 执行推测解码 draft 分支核心计算逻辑
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            # 获取当前全局前向上下文
            forward_context = get_forward_context()
            # 全图模式且未处于图捕获阶段，更新图运行时参数
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

    # 构造首轮模型输入字典：预处理上下文 KV 并组装 input_ids / positions
    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,  # 本轮有效输入 token 总数
    ) -> dict[str, Any]:
        # 获取当前 DFlash 记录的上下文 token 数量
        num_context = self._dflash_num_context

        # 预计算上下文 K/V 并存入 KV Cache，传入隐态、位置、slot 映射
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context],
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )

        # 组装模型标准输入字典：token id、位置编码、无外部嵌入
        return dict(
            input_ids=self.input_ids[:num_input_tokens], positions=self.positions[:num_input_tokens], inputs_embeds=None
        )

    # 多模态检测占位函数：DFlash 推测解码暂不支持多模态，直接空实现
    def _raise_if_multimodal(self):
        pass  


# 动态num_spec_tokens
class AscendDflash2Proposer(AscendEagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,  # vLLM 全局配置对象，包含批大小、序列长度、模型参数等
        device: torch.device,     # 计算设备（昇腾 NPU 设备）
        runner=None,               # vLLM 运行时管理器，负责调度、KV Cache、批处理等
    ):
        super().__init__(
            vllm_config,
            device,
            runner=runner,
        )
