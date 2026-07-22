# 类型导入
from typing import Any, cast
# 数值计算库
import numpy as np
# 深度学习张量库
import torch
# vLLM 向上取整工具函数
from vllm.utils.math_utils import cdiv
# vLLM V1 注意力后端常量：PAD_SLOT_ID代表无效KV槽位，多卡CP并行时不负责该token的卡填充此值
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
# vLLM V1 KV缓存分组描述结构体，用于多分组独立KV缓存
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
# vLLM V1 CP并行工具：获取全局上下文并行总卡数
from vllm.v1.worker.cp_utils import get_total_cp_world_size

# 导入昇腾底层基础块表父类，重写适配CP并行slot映射逻辑
from vllm_ascend.worker.block_table import BlockTable as AscendBlockTable
# 导入昇腾底层多分组块表父类，用于多KV缓存分组场景
from vllm_ascend.worker.block_table import MultiGroupBlockTable as AscendMultiGroupTable


# 单KV分组块表实现，继承昇腾原生块表，重写slot_mapping计算逻辑适配CP上下文并行
class BlockTable(AscendBlockTable):
    # 对外统一入口：计算当前批所有token的KV槽位映射slot_mapping
    def compute_slot_mapping(self, *args: Any) -> None:
        # 标准化输入参数，统一输出(req_indices每条token归属请求ID, positions每条token在请求内的全局逻辑位置)
        req_indices, positions = self._normalize_slot_mapping_inputs(*args)
        # 调用CPU侧numpy核心计算函数，生成slot_mapping
        self._compute_slot_mapping_numpy(req_indices, positions)

    # 核心CPU numpy计算函数：将token逻辑位置映射为NPU显存KV物理slot地址，区分单卡/多卡CP并行两条分支
    def _compute_slot_mapping_numpy(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        # 获取当前批总token数量
        num_tokens = positions.shape[0]
        # 无token空批，清空slot_mapping显存缓冲区直接返回
        if num_tokens == 0:
            self.slot_mapping.copy_to_gpu(0)
            return

        # 判断是否开启CP上下文并行（DCP分布式CP + PCP流水线CP 总并行卡数大于1）
        if self.dcp_world_size * self.pcp_world_size > 1:
            # 虚拟块大小：单逻辑块融合所有并行卡对应物理块，虚拟块容纳所有CP卡同偏移token
            virtual_block_size = self.block_size * self.dcp_world_size * self.pcp_world_size
            # 计算每个token归属的虚拟块下标
            logical_block_idx = positions // virtual_block_size
            # 换算token在一维扁平化block_table数组中的索引
            block_table_indices = self._get_block_table_indices(req_indices, logical_block_idx)
            # 根据索引取出token对应KV物理块编号
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # token在当前虚拟块内部的偏移量
            virtual_block_offsets = positions % virtual_block_size
            # 计算当前NPU卡全局rank编号（DCP维度+PCP维度合并rank）
            current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
            # 掩码数组：True代表当前卡负责存储该token的KV，False不负责填充PAD_SLOT_ID
            mask = (
                virtual_block_offsets // self.cp_kv_cache_interleave_size % (self.dcp_world_size * self.pcp_world_size)
                == current_rank
            )
            # 重映射虚拟块偏移到当前卡本地物理块内真实slot偏移
            block_offsets = (
                virtual_block_offsets
                // (self.dcp_world_size * self.pcp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # 标准slot计算公式：物理块号*单块token容量 + 块内偏移 = 显存KV槽位ID
            slot_mapping = block_numbers * self.block_size + block_offsets
            # 根据掩码填充slot_mapping：当前卡负责的token填真实slot，不负责的填充无效PAD标识
            self.slot_mapping.np[:num_tokens] = np.where(mask, slot_mapping, PAD_SLOT_ID)
        # 单卡无CP并行分支，无分片逻辑，简化slot计算
        else:
            # 按物理块大小计算token归属的物理块下标
            logical_block_idx = positions // self.block_size
            # 换算block_table一维数组索引
            block_table_indices = self._get_block_table_indices(req_indices, logical_block_idx)
            # 取出对应物理块编号
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # token在物理块内的偏移量
            block_offsets = positions % self.block_size
            # 原地计算slot id，写入CPU侧slot_mapping数组
            np.add(block_numbers * self.block_size, block_offsets, out=self.slot_mapping.np[:num_tokens])

        # 将CPU算好的slot_mapping拷贝至NPU显存，供给注意力算子读取KV缓存
        self.slot_mapping.copy_to_gpu(num_tokens)

    # 辅助函数：计算token在扁平化block_table一维数组中的下标
    def _get_block_table_indices(self, req_indices, logical_block_idx):
        # row_stride：单个请求在block_table中占用的行长度（单请求最大块数*每个物理块占用表项）
        row_stride = self.max_num_blocks_per_req * self.blocks_per_phys_block
        # 请求起始偏移 + 块内下标 = 一维数组索引
        return req_indices * row_stride + logical_block_idx

    # 输入参数标准化：兼容2参数(Draft推测解码)/3参数(Target正常解码)两种调用格式，统一输出numpy数组(req_indices, positions)
    def _normalize_slot_mapping_inputs(self, *args) -> tuple[np.ndarray, np.ndarray]:
        # 2参数模式：Draft模型推测解码专用，直接传入req_indices、positions
        if len(args) == 2:
            req_indices, positions = args
            return self._to_numpy(req_indices), self._to_numpy(positions)

        # 3参数模式：Target主解码流程，入参(num_reqs批请求总数, query_start_loc请求token偏移数组, positions所有token位置)
        if len(args) == 3:
            num_reqs, query_start_loc, positions = args
            # 转换请求偏移数组，截断到有效请求长度+1
            query_start_loc_np = self._to_numpy(query_start_loc)[: num_reqs + 1]
            # 转换所有token位置数组
            positions_np = self._to_numpy(positions)
            # 差分计算每个请求包含的token数量
            counts = np.diff(query_start_loc_np)
            # 按每个请求token数量重复请求ID，生成与positions等长的req_indices数组
            req_indices_np = np.repeat(np.arange(num_reqs, dtype=np.int64), counts)
            # 长度校验：请求ID数组长度必须等于总token数，防止参数不匹配报错
            if req_indices_np.shape[0] != positions_np.shape[0]:
                raise ValueError(
                    "query_start_loc and positions describe different token counts: "
                    f"{req_indices_np.shape[0]} != {positions_np.shape[0]}"
                )
            return req_indices_np, positions_np

        # 入参数量非法，抛出类型异常
        raise TypeError("compute_slot_mapping expects either 2 or 3 positional arguments")

    # 静态工具方法：统一将输入张量/数组转为CPU侧int64 numpy数组，禁止NPU设备张量输入
    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        # 原生numpy数组：直接转int64不拷贝
        if isinstance(value, np.ndarray):
            return value.astype(np.int64, copy=False)
        # torch张量分支
        if isinstance(value, torch.Tensor):
            # 强制校验必须是CPU张量，NPU张量会引入D2H拷贝与NPU小算子开销，昇腾310P不支持
            if value.device.type != "cpu":
                raise TypeError(
                    "310P slot mapping must be computed from CPU req_indices/positions; "
                    "device tensor inputs would require unsupported NPU arithmetic or D2H"
                )
            # 张量脱离计算图，转为numpy int64数组
            return value.detach().numpy().astype(np.int64, copy=False)
        # 列表/标量等其他类型：统一转为int64 numpy
        return np.asarray(value, dtype=np.int64)


# 多分组块表，支持多套独立KV缓存（MoE/多模型/多尺寸block场景），内部管理多个单组BlockTable实例
class MultiGroupBlockTable(AscendMultiGroupBlockTable):
    # 多分组块表构造函数，接收num_speculative_tokens透传给每个子块表，用于推测解码KV扩容
    def __init__(
        self,
        max_num_reqs: int,                  # 服务支持最大并发请求数
        max_model_len: int,                 # 模型上下文最大长度
        max_num_batched_tokens: int,        # 单批最大token数量
        pin_memory: bool,                   # 是否锁页内存，加速CPU-NPU拷贝
        device: torch.device,               # NPU设备对象
        block_sizes: list[int],             # 每个KV分组对应的block块尺寸列表
        num_speculative_tokens: int = 0,    # 推测解码单次预生成候选token数量，0关闭spec
        max_num_blocks: list[int] | None = None, # 每个分组单请求最大占用块数
        kernel_sizes: list[list[int]] | None = None, # 各分组注意力内核尺寸配置
        cp_kv_cache_interleave_size: int = 1, # CP并行KV分片交错粒度
        kv_cache_groups: list[KVCacheGroupSpec] | None = None, # 多分组独立KV缓存池描述
    ) -> None:
        # 内核尺寸为空时，填充默认[[0]]占位
        if kernel_sizes is None:
            kernel_sizes = [[0]] * len(block_sizes)
        # 仅传入1套内核配置，但存在多个KV分组时，复制配置适配所有分组
        elif len(kernel_sizes) == 1 and len(block_sizes) > 1:
            kernel_sizes = kernel_sizes * len(block_sizes)
        # 内核配置数量与分组数量不匹配，抛出异常
        elif len(kernel_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match block_sizes length ({len(block_sizes)})"
            )

        # 未传入单请求最大块数时，根据模型最大长度、CP并行卡数自动计算
        if max_num_blocks is None:
            total_cp_world_size = get_total_cp_world_size()
            # 向上取整：单请求最大块数 = 最大上下文长度 / (单块大小 * CP总并行卡数)
            max_num_blocks = [cdiv(max_model_len, block_size * total_cp_world_size) for block_size in block_sizes]

        # 最大块数量列表长度与分组数量不匹配，抛出异常
        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) must match block_sizes length ({len(block_sizes)})"
            )

        # 存在多组独立KV缓存池：循环初始化每个分组专属BlockTable，透传num_speculative_tokens
        if kv_cache_groups is not None:
            self.block_tables = [
                BlockTable(
                    block_size,                # 当前分组块大小
                    max_num_reqs,              # 全局最大并发请求
                    max_num_blocks_per_req,    # 当前分组单请求最大块数
                    max_num_batched_tokens,    # 全局批最大token
                    pin_memory,                # 锁页内存开关
                    device,                    # NPU设备
                    kernel_size_list,          # 当前分组内核尺寸
                    cp_kv_cache_interleave_size, # CP分片交错粒度
                    num_speculative_tokens,    # 推测解码候选token数，用于KV空间扩容
                    kv_cache_group,            # 当前分组独立KV缓存池配置
                )
                for block_size, kernel_size_list, max_num_blocks_per_req, kv_cache_group in zip(
                    block_sizes, kernel_sizes, max_num_blocks, kv_cache_groups
                )
            ]
        # 无独立KV分组，所有分组共享全局KV缓存池，初始化子BlockTable
        else:
            self.block_tables = [
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_size_list,
                    cp_kv_cache_interleave_size,
                    num_speculative_tokens, # 透传推测解码参数给底层块表，预分配多token KV空间
                )
                for block_size, kernel_size_list, max_num_blocks_per_req in zip(
                    block_sizes, kernel_sizes, max_num_blocks
                )
            ]

    # 通用slot_mapping计算入口：遍历所有KV分组，分别计算每组的KV槽位映射
    def compute_slot_mapping(
        self,
        num_reqs_or_req_indices: int | np.ndarray | torch.Tensor, # 3参模式为总请求数；2参模式为req_indices数组
        query_start_loc_or_positions: np.ndarray | torch.Tensor, # 3参模式为请求偏移数组；2参模式为positions
        positions: np.ndarray | torch.Tensor | None = None,      # 3参模式有效，存储全部token位置
        positions_compressed_list: list[np.ndarray] | None = None, # 稀疏批压缩token位置列表，多分组独立压缩
        req_indices_compressed_list: list[np.ndarray] | None = None, # 稀疏批压缩请求ID列表
    ) -> None:
        # 遍历所有KV分组块表
        for i, block_table_base in enumerate(self.block_tables):
            # 强制类型转换为自定义BlockTable
            block_table = cast(BlockTable, block_table_base)
            # 稀疏压缩输入模式：取当前分组压缩后的req、positions调用2参compute_slot_mapping
            if positions_compressed_list is not None and req_indices_compressed_list is not None:
                block_table.compute_slot_mapping(
                    req_indices_compressed_list[i],
                    positions_compressed_list[i],
                )
            # positions为空，代表传入3参数格式(num_reqs, query_start_loc)
            elif positions is None:
                block_table.compute_slot_mapping(
                    num_reqs_or_req_indices,
                    query_start_loc_or_positions,
                )
            # 完整3参数标准主解码流程
            else:
                block_table.compute_slot_mapping(
                    num_reqs_or_req_indices,
                    query_start_loc_or_positions,
                    positions,
                )

    # 推测解码Draft模型专用slot映射计算接口，固定2参数模式，批量处理num_speculative_tokens个候选token
    def compute_slot_mapping_draft(
        self,
        req_indices: np.ndarray,                                 # Draft批每条token归属请求ID
        positions: np.ndarray,                                   # Draft批预生成候选token的逻辑位置
        positions_compressed_list: list[np.ndarray] | None = None, # 稀疏压缩位置列表
        req_indices_compressed_list: list[np.ndarray] | None = None, # 稀疏压缩请求ID列表
    ) -> None:
        # 遍历全部KV分组块表
        for i, block_table_base in enumerate(self.block_tables):
            block_table = cast(BlockTable, block_table_base)
            # 稀疏压缩输入分支，取当前分组压缩数据调用2参slot计算
            if positions_compressed_list is not None and req_indices_compressed_list is not None:
                block_table.compute_slot_mapping(
                    req_indices_compressed_list[i],
                    positions_compressed_list[i],
                )
            # 标准Draft 2参数调用，一次性计算num_speculative_tokens个候选token的KV槽位
            else:
                block_table.compute_slot_mapping(req_indices, positions)