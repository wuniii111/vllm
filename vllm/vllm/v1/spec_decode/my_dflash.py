# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# DFlash推测解码Proposer实现：基于并行预填充多候选token的推测解码提案器，依托triton核实现输入打包

from typing import Any
# 通用类型注解

import torch
# PyTorch张量与CUDA运算依赖
from typing_extensions import override
# 子类重写装饰器注解

from vllm.config import VllmConfig
# vLLM全局配置类，包含推测解码、模型、硬件配置
from vllm.forward_context import set_forward_context
# 前向传播上下文管理器，用于CUDA Graph捕获、分布式token数标记
from vllm.logger import init_logger
# vLLM日志初始化工具
from vllm.triton_utils import triton
# triton编译工具、算力工具函数（向上取2次幂、分块计算等）
from vllm.v1.attention.backend import CommonAttentionMetadata
# V1版本通用注意力元数据：包含seq_len、slot_mapping、block_table、query_start_loc等KV缓存信息
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
# 推测解码提案器基类，所有spec提案器的父类
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel
# DFlash专用triton融合核：批量拼接上下文+查询侧输入id、位置、slot映射

logger = init_logger(__name__)
# 初始化本模块日志实例


# 新增 ###
def update_block_size_from_acceptance(
    current_block_size: int,
    avg_acceptance: float,
    min_block_size: int = 2,
    max_block_size: int = 16,
) -> int:
    """Adjust the DFlash speculative block size using a simple acceptance rule.

    The policy mirrors the requested heuristic:
    - increase when acceptance is above 0.85
    - decrease when acceptance is below 0.5
    - keep the current value otherwise
    """
    if current_block_size <= 0:
        return max(min_block_size, min(max_block_size, min_block_size))

    if avg_acceptance > 0.85:
        return min(max_block_size, current_block_size + 1)
    if avg_acceptance < 0.5:
        return max(min_block_size, current_block_size - 1)
    return current_block_size


"""
proposer
给定当前上下文，快速生成若干候选 token（draft tokens），交给大模型验证
"""


class DFlashProposer(SpecDecodeBaseProposer):
    """DFlash并行推测解码提案器，核心逻辑：
    1. target模型前向得到上下文hidden_state，预计算存入KV缓存作为Context-KV
    2. 每条请求构造1个基准token + N个spec候选token作为Query，非因果交叉注意力查询上文KV
    3. Triton一键打包全量输入，适配CUDA Graph静态显存布局
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        # 构造函数：初始化DFlash专属缓存buffer、配置参数
        # 强校验：必须开启推测配置且解码方式为dflash
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        # 调用父类构造：透传配置，开启透传hidden_state给子模型标记，绑定推理runner
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # 单请求查询token：1个真实next_token + num_speculative个draft候选token，全局最大查询总token数
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # 总位置容量：上下文原有token + 新增查询token，用于positions索引预分配
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # ========== 分离上下文/查询两块显存buffer：分离是为固定查询buffer地址，兼容CUDA Graph捕获 ==========
        # 上下文slot映射Buffer：上下文token对应的KV缓存slot编号，固定最大上下文token长度
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        # 查询侧slot映射Buffer：所有query（基准+spec候选）对应的slot，预分配最大查询token容量
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        # 上下文位置编码Buffer：上下文每个token的position索引
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        # 查询侧位置编码Buffer：所有query token的position索引
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        # 预生成连续整数数组：用于快速生成query_start_loc分片索引，int32节省显存
        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # DFlash使用输入Embedding拼接mask占位token，预留并行draft隐状态张量，运行时动态初始化
        self.parallel_drafting_hidden_state_tensor = None
        self._dflash_block_size_min = 2  ###
        self._dflash_block_size_max = 16  ###

    @override
    def _raise_if_multimodal(self):
        """重写基类多模态校验：DFlash原生支持多模态Qwen3.5，关闭多模态异常拦截
        备注：多模态输入全链路未经过完备测试
        """
        # 空实现，跳过多模态报错
        pass

    def _maybe_update_block_size_from_acceptance(
        self, batch_size: int, num_rejected_tokens_gpu: torch.Tensor | None
    ) -> None:
        if batch_size <= 0 or self.num_speculative_tokens <= 0:
            return

        if self.block_size <= 0:
            self.block_size = self.vllm_config.cache_config.block_size

        if num_rejected_tokens_gpu is None:
            return

        total_rejected = int(num_rejected_tokens_gpu.sum().item())
        total_spec_tokens = batch_size * self.num_speculative_tokens
        avg_acceptance = max(
            0.0,
            1.0 - (total_rejected / total_spec_tokens),
        )
        self.block_size = update_block_size_from_acceptance(
            current_block_size=self.block_size,
            avg_acceptance=avg_acceptance,
            min_block_size=self._dflash_block_size_min,
            max_block_size=self._dflash_block_size_max,
        )

    @override  # 新增 重写父类方法，编译器会帮你校验：方法名、参数列表、返回值是否和父类完全一致
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        """首轮前向输入组装：DFlash核心入口
        逻辑：上下文KV来自target模型输出hidden_state，Query由next_token+spec_mask_token组成，triton核一次性填充全部输入buffer
        返回：总查询token数、采样索引张量、重构后的注意力元数据
        """
        # 当前推理batch请求数
        batch_size = cad.batch_size()
        # 上下文原始token总数量（target输入全量token）
        num_context = target_token_ids.shape[0]
        # 单请求查询token：1个真实落地token + N个推测候选token
        num_query_per_req = 1 + self.num_speculative_tokens
        # 全局所有请求合计query总token
        num_query_total = batch_size * num_query_per_req

        # 缓存上下文token计数，后续build_model_inputs_first_pass使用
        self._dflash_num_context = num_context

        # 根据上一轮推测解码接受率动态调整 DFlash 使用的 block size。
        # 这只影响 proposer 侧的 slot 映射与打包逻辑，不改变底层 KV cache
        # allocator 的全局 block size 配置。
        self._maybe_update_block_size_from_acceptance(
            batch_size=batch_size,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )

        # 保存target模型输出隐状态，后续预计算上下文KV使用；上下文预处理不走CUDA Graph无需拷贝至固定buffer
        self._dflash_hidden_states = target_hidden_states

        # 预分配采样索引张量：仅spec候选token需要采样，长度=batch*spec_token数
        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Triton核分块配置：单请求最大上下文+查询长度，块大小向上取2次幂且不超过256
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        # 计算需要多少个triton块
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        # triton网格：[batch维度, 单请求分块数]
        grid = (batch_size, num_blocks)

        # 判断是否存在被拒绝作废token（上一轮校验失败丢弃的spec token）
        has_num_rejected = num_rejected_tokens_gpu is not None
        # 启动Triton融合内核：一次性填充input_ids、上下文/查询positions、slot映射、采样索引
        copy_and_expand_dflash_inputs_kernel[grid](
            # 输入参数：上游target输出token、基准token位置
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # 输出参数：各类预分配buffer指针
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # KV缓存块表信息：block_table张量+步长
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # 请求起始位置、作废token计数
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # 常量超参：mask占位token id、KV页大小、单请求查询数、spec候选数、原始上下文token数
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            # triton编译期常量
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        # 截取有效查询侧slot映射（只取真实占用长度）
        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        # 生成新query分片起始位置：每条请求固定num_query_per_req个查询token
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req

        # 修正有效上下文长度：存在作废token时，从原始seq_len减去作废token数量，过滤无效上下文
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        # 更新CPU侧序列长度上界（用于内存预分配）
        new_seq_lens_cpu_upper_bound = (
            cad.seq_lens_cpu_upper_bound + num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        # 构造DFlash专属注意力元数据：非因果注意力，Query查全量上下文KV
        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu_upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=cad.max_seq_len + num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=False,  # DFlash必须关闭因果掩码，Query可见全部上下文KV
        )

        # 返回总查询token、候选采样索引、新注意力元数据
        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """空跑预热：CUDA Graph捕获前做占位前向，预分配显存、初始化算子，DFlash独有特性：
        1. 仅单次模型前向（并行draft一次性生成所有spec候选）
        2. 上下文KV提前预计算，不走模型常规KV逻辑
        3. 仅Query侧token流入draft模型做注意力计算
        """
        # 实际使用查询token数，不超过预分配最大上限
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        # 计算padding补齐后token数、DP分片token数、CUDA Graph运行标记
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # 拼接注意力层slot_mapping字典，仅Query侧slot传入模型，上下文KV提前写入缓存
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # 截取有效上下文位置buffer
        context_positions = self._context_positions_buffer[:num_tokens]
        # dummy预热使用预留隐状态buffer充当上下文hidden_state
        context_states = self.hidden_states[:num_tokens]

        # 预热：预计算上下文KV（Norm+RoPE+KV投影+写入KV缓存），提前落库上下文KV
        self.model.precompute_and_store_context_kv(context_states, context_positions)
        # 绑定前向上下文，开启CUDA Graph捕获环境
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            # draft模型前向，仅输入Query侧input_ids与positions，不传入embeds
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        """组装首轮模型入参：DFlash在模型前向之前，先把上下文KV预存入缓存
        返回：模型输入字典 + 有效输入token数
        """
        # 读取之前缓存的上下文总token数量
        num_context = self._dflash_num_context

        # 提前使用target输出hidden_state+上下文pos+slot，预计算并落地上下文KV到缓存
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # target模型输出隐状态 [num_context, hidden_dim]
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        # 构造draft模型入参：仅传入Query侧id与pos，embeds为空由模型内部查表
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        """分层构造每层注意力元数据，强制校验所有注意力层关闭因果掩码(causal=False)
        DFlash依赖非因果注意力查询全量上下文，因果模式直接抛异常
        """
        # 调用父类生成基础分层注意力元数据
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        # 遍历所有注意力层，强制校验causal标记为False
        for layer_name, attn_metadata in per_layer.items():
            assert getattr(attn_metadata, "causal", None) is False, (
                f"Attention metadata for layer {layer_name} does not have"
                " non-causal support, which is required for DFlash."
                " Consider using a different attention backend, such as FlashAttention."
            )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        """读取模型配置，确认是否启用辅助隐状态输出，DFlash默认开启aux hidden_state
        从模型hf_config.dflash_config读取use_aux_hidden_state配置，无配置则默认True
        """
        # 默认开启辅助隐状态
        use_aux_hidden_state = True
        # 从draft模型HF配置读取dflash自定义配置
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        # 配置存在则读取开关，不存在沿用默认True
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state