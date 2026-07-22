import numpy as np
import torch
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
from vllm.v1.worker.cp_utils import get_total_cp_world_size


# 单请求KV块管理核心类，昇腾vLLM V1 KV缓存BlockTable底层基类
class BlockTable:
    # 构造函数：初始化块表、CP并行配置、KV缓存缓冲区、适配推测解码num_speculative_tokens扩容
    def __init__(
        self,
        block_size: int,                          # 物理KV块单块容纳token数量
        max_num_reqs: int,                        # 全局最大并发请求上限
        max_num_blocks_per_req: int,              # 单个请求最多占用物理KV块数量
        max_num_batched_tokens: int,              # 单批推理最大token总数
        pin_memory: bool,                         # 是否启用锁页内存，加速CPU<->NPU数据拷贝
        device: torch.device,                     # 当前绑定的昇腾NPU设备
        kernel_sizes: list[int] | None = None,    # 注意力内核切分尺寸，用于混合逻辑块拆分物理块
        cp_kv_cache_interleave_size: int = 1,      # CP上下文并行KV缓存交错分片粒度
        num_speculative_tokens: int = 0,          # 推测解码单次预生成候选token数量，用于扩容块表
        kv_cache_group: KVCacheGroupSpec = None,   # 当前分组独立KV缓存池配置对象
    ):
        # 保存全局最大并发请求数
        self.max_num_reqs = max_num_reqs
        # KV压缩倍率，默认1（无KV压缩）
        compress_ratio = 1
        # 读取KV分组配置中的KV缓存压缩倍率
        if (
            kv_cache_group is not None
            and hasattr(kv_cache_group, "kv_cache_spec")
            and hasattr(kv_cache_group.kv_cache_spec, "compress_ratio")
        ):
            compress_ratio = kv_cache_group.kv_cache_spec.compress_ratio
        # 向上取整计算压缩后单请求所需最大块数，最小为1
        max_num_blocks_per_req = max(cdiv(max_num_blocks_per_req, compress_ratio), 1)
        # 保存压缩修正后的单请求最大块数
        self.max_num_blocks_per_req = max_num_blocks_per_req
        # 单批最大token数
        self.max_num_batched_tokens = max_num_batched_tokens
        # 锁页内存开关
        self.pin_memory = pin_memory
        # NPU设备对象
        self.device = device
        # 原始物理KV块尺寸
        self.physical_block_size = block_size

        try:
            # 获取PCP流水线上下文并行通信组
            self.pcp_world_size = get_pcp_group().world_size
            # 当前卡PCP组内rank，无PCP并行则置0
            self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_world_size > 1 else 0
            # 获取DCP分布式上下文并行通信组总卡数
            self.dcp_world_size = get_dcp_group().world_size
            # 当前卡DCP组内rank编号
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # 测试环境未初始化DCP/PCP并行组时捕获异常，降级为单卡并行
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.pcp_world_size = 1
            self.pcp_rank = 0

        # 分支1：无内核切分（kernel_sizes为空或[0]），物理块=逻辑块，不做混合块拆分
        if kernel_sizes is None or kernel_sizes == [0]:
            self.block_size = block_size                # 逻辑块尺寸=物理块尺寸
            self.logical_block_size = block_size        # 逻辑块单块token容量
            self.blocks_per_phys_block = 1             # 1个物理块对应1个逻辑块，无拆分
            self.use_hybrid_blocks = False              # 关闭混合逻辑块拆分模式
        # 分支2：开启内核切分，将大物理块拆分为多个小逻辑块适配注意力算子
        else:
            # 遍历内核尺寸，找到第一个能整除物理块大小的合法切分尺寸
            selected_kernel_size = None
            for kernel_size in kernel_sizes:
                if kernel_size > 0 and self.physical_block_size % kernel_size == 0:
                    selected_kernel_size = kernel_size
                    break
            # 所有内核尺寸都无法整除物理块，抛出参数异常
            if selected_kernel_size is None:
                raise ValueError(
                    f"None of the kernel sizes {kernel_sizes} can divide "
                    f"physical block size {self.physical_block_size} evenly"
                )
            # 使用选中的切分尺寸作为逻辑块大小
            self.block_size = selected_kernel_size
            self.logical_block_size = selected_kernel_size
            # 单个物理块可拆分成的逻辑块总数
            self.blocks_per_phys_block = self.physical_block_size // self.logical_block_size
            # 物理块可拆分为多逻辑块时，开启混合块模式
            if self.blocks_per_phys_block > 1:
                self.use_hybrid_blocks = True
            else:
                self.use_hybrid_blocks = False

        # 计算块表单行逻辑长度：混合块模式需要乘拆分倍数扩容
        if self.use_hybrid_blocks:
            logical_table_size = max_num_blocks_per_req * self.blocks_per_phys_block
        else:
            logical_table_size = max_num_blocks_per_req


        # CP 上下文并行：沿序列长度 T（token 维度） 切分 KV 缓存，把同一条请求的上下文 token 分散存储在多张 NPU/GPU 卡上
        # 块表横向扩容系数：CP并行场景下额外叠加num_speculative_tokens容纳推测多候选token
        # 基础扩张系数，默认单卡无CP并行时只需要1倍容量
        duplicate_size = 1
        # 判断：是否开启了PCP+DCP组合上下文并行（多卡CP分片KV缓存）
        if self.pcp_world_size * self.dcp_world_size > 1:
            # 开启CP并行时，扩容系数叠加推测解码候选token数量
            duplicate_size += num_speculative_tokens
        # 创建CPU+NPU双端块表缓冲区，shape=[最大请求数*扩容系数, 单行逻辑块长度]，存储int32物理块ID
        self.block_table = self._make_buffer(max_num_reqs * duplicate_size, logical_table_size, dtype=torch.int32)
        # 数组：记录每个请求当前已占用的有效块数量，用于块表读写边界控制
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        # 创建slot_mapping槽位映射缓冲区，预留冗余长度适配CP并行多卡偏移
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens + 2 * self.pcp_world_size * self.max_num_reqs, dtype=torch.int32
        )

        # 保存注意力内核配置列表
        self.kernel_sizes = kernel_sizes
        # 保存CP并行KV分片交错粒度
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    # 追加物理块ID到指定请求行末尾，用于增量追加KV块（预填充/解码阶段扩容上下文）
    def append_row(
        self,
        block_ids,          # 待追加的物理KV块ID列表
        row_idx: int,       # 目标请求的行下标
    ) -> None:
        # 空块列表直接返回，无需操作
        if not block_ids:
            return
        # 转换为numpy数组便于批量处理
        block_ids = np.array(block_ids)
        # 混合块模式：将物理块ID拆分为多个连续逻辑块ID
        if self.use_hybrid_blocks:
            block_ids = self._convert_physical_to_logical_blocks(block_ids)
        # 待写入块的总数量
        num_blocks = len(block_ids)
        # 获取该行当前已占用块的末尾偏移
        start = self.num_blocks_per_row[row_idx]
        # 将转换后的逻辑块ID写入块表对应行的空闲区间
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids
        # 更新该行有效块计数
        self.num_blocks_per_row[row_idx] += num_blocks

    # 重置指定请求行，覆盖写入全新KV块列表
    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        # 先清空该行原有有效块计数
        self.num_blocks_per_row[row_idx] = 0
        # 调用追加接口写入新块ID
        self.append_row(block_ids, row_idx)

    # 清空指定请求行的所有KV块数据，回收KV缓存
    def clear_row(self, row_idx: int) -> None:
        # 获取该行有效块数量
        num_blocks = self.num_blocks_per_row[row_idx]
        # 存在有效块则置0清空块表区间
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        # 重置该行块计数
        self.num_blocks_per_row[row_idx] = 0

    # 将源请求行的KV块完整拷贝至目标请求行（请求迁移/缓存复用场景）
    def move_row(self, src: int, tgt: int) -> None:
        # 获取源请求有效块总数
        num_blocks = self.num_blocks_per_row[src]
        # 块数据拷贝到目标行头部
        self.block_table.np[tgt, :num_blocks] = self.block_table.np[src, :num_blocks]
        # 同步目标行有效块计数
        self.num_blocks_per_row[tgt] = num_blocks

    # 交换两个请求行的全部KV块数据（请求调度重排场景）
    def swap_row(self, src: int, tgt: int) -> None:
        # 分别取出两行有效块数量
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        # 交换两行的块计数
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src
        # 批量交换块表两行数据
        self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]

    # Target主解码流程slot映射计算：调用昇腾NPU内核并行生成slot_mapping
    def compute_slot_mapping(
        self,
        num_reqs: int,                          # 当前批有效请求总数
        query_start_loc: torch.Tensor,          # 请求token偏移数组，[num_reqs+1]
        positions: torch.Tensor,                # 批内所有token全局逻辑位置
    ) -> None:
        # 当前CP并行总卡数=PCP流水线卡数 * DCP分布式卡数
        num_tokens = positions.shape[0]
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        # 合并PCP+DCP的全局rank编号
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        # 调用昇腾自定义NPU核函数并行计算slot_mapping，传入并行、块尺寸、填充标识等超参
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    # 推测解码Draft模型专用slot映射计算，一次性处理num_speculative_tokens个预生成候选token
    def compute_slot_mapping_draft(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        # 注释示例：req_indices映射规则，多token场景下请求ID重复匹配对应虚拟块偏移
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, 2*K, 2*K, 2*K+1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.

        # 开启CP多卡并行分支，处理KV分片交错存储寻址逻辑
        if self.dcp_world_size * self.pcp_world_size > 1:
            # DCP交错存储说明：token下标i的KV缓存固定存储在 dcp_rank = i % pcp_world_size 的卡上
            # 虚拟块大小 = 单逻辑块尺寸 × 全部CP并行卡总数，融合多卡同偏移token
            virtual_block_size = self.block_size * self.dcp_world_size * self.pcp_world_size
            # 关键：混合块模式下positions是逻辑块坐标，计算token归属虚拟块下标
            logical_block_idx = positions // virtual_block_size
            # 换算token在一维扁平化块表数组的索引，叠加混合块拆分扩容倍数
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx
            )
            # 根据索引取出token对应的逻辑块ID
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # token在虚拟块内部的偏移量
            virtual_block_offsets = positions % virtual_block_size
            # 计算当前卡全局CP rank
            self.current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
            # 布尔掩码：True=当前NPU卡负责存储该token的KV，False=不负责填充PAD(-1)
            mask = (
                virtual_block_offsets // self.cp_kv_cache_interleave_size % (self.dcp_world_size * self.pcp_world_size)
                == self.current_rank
            )
            # 将虚拟块交错偏移重映射到当前卡本地逻辑块内部真实偏移
            block_offsets = (
                virtual_block_offsets
                // (self.dcp_world_size * self.pcp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # 标准slot计算公式：逻辑块ID × 单块token容量 + 块内偏移 = KV显存槽位ID
            slot_mapping = block_numbers * self.block_size + block_offsets
            # 写入slot_mapping缓冲区，非当前卡token填充-1无效标识
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(mask, slot_mapping, -1)
        # 单卡无CP并行简化分支，无分片掩码逻辑
        else:
            # 校验内核与块尺寸配置合法性
            assert self.kernel_sizes is not None
            assert self.block_size == self.kernel_sizes[0]
            # 关键：混合块模式positions为逻辑坐标，计算token归属逻辑块下标
            logical_block_idx = positions // self.block_size
            # 换算一维扁平化块表索引，叠加混合块扩容倍数
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req * self.blocks_per_phys_block + logical_block_idx
            )
            # 取出对应逻辑块ID
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # token在逻辑块内部偏移
            block_offsets = positions % self.block_size
            # 原地计算slot id写入CPU侧slot_mapping数组
            np.add(block_numbers * self.block_size, block_offsets, out=self.slot_mapping.np[: req_indices.shape[0]])
            # 将CPU算好的slot映射拷贝至NPU显存
            self.slot_mapping.copy_to_gpu(req_indices.shape[0])

    # 将CPU侧更新后的块表同步拷贝至NPU显存，仅同步前num_reqs行有效数据
    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    # 清空整块表所有数据，CPU+NPU双缓冲区全部置0
    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table.cpu.fill_(0)

    # 工具函数：物理块ID批量转换为拆分后的连续逻辑块ID（混合块模式专用）
    def _convert_physical_to_logical_blocks(self, physical_blocks: np.ndarray) -> np.ndarray:
        """Convert physical block IDs to logical block IDs."""
        # 非混合块模式直接原样返回物理块ID
        if not self.use_hybrid_blocks:
            return physical_blocks
        # 存储转换后逻辑块ID列表
        logical_blocks: list[int] = []
        # 遍历每个物理块，拆分为多个连续逻辑块
        for phys_block in physical_blocks:
            # 物理块基准逻辑ID = 物理块编号 × 单物理块拆分逻辑块数量
            base_logical = phys_block * self.blocks_per_phys_block
            # 追加该物理块拆分出的全部连续逻辑块ID
            logical_blocks.extend(range(base_logical, base_logical + self.blocks_per_phys_block))
        # 转为int32 numpy数组返回
        return np.array(logical_blocks, dtype=np.int32)

    # 获取NPU设备侧块表张量
    def get_device_tensor(self) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu

    # 获取CPU主机侧块表张量
    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    # 获取CPU侧numpy数组视图，用于numpy批量计算slot映射
    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    # 内部缓冲区创建工具：生成同时持有CPU锁页内存+NPU显存的双端CpuGpuBuffer
    def _make_buffer(self, *size: int | torch.SymInt, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size, dtype=dtype, device=self.device, pin_memory=self.pin_memory)


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        num_speculative_tokens: int = 0,
        max_num_blocks: list[int] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        kv_cache_groups: KVCacheGroupSpec = None,
    ) -> None:
        if kernel_sizes is None:
            kernel_sizes = [[0]] * len(block_sizes)
        # Ensure kernel_sizes matches block_sizes length
        elif len(kernel_sizes) == 1 and len(block_sizes) > 1:
            kernel_sizes = kernel_sizes * len(block_sizes)
        elif len(kernel_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match block_sizes length ({len(block_sizes)})"
            )

        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [cdiv(max_model_len, block_size * total_cp_world_size) for block_size in block_sizes]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) must match block_sizes length ({len(block_sizes)})"
            )

        # Use zip to pair block_sizes with kernel_sizes one-to-one
        if kv_cache_groups is not None:
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
                    num_speculative_tokens,
                    kv_cache_group,
                )
                for block_size, kernel_size_list, max_num_blocks_per_req, kv_cache_group in zip(
                    block_sizes, kernel_sizes, max_num_blocks, kv_cache_groups
                )
            ]
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
                    num_speculative_tokens,
                )
                for block_size, kernel_size_list, max_num_blocks_per_req in zip(
                    block_sizes, kernel_sizes, max_num_blocks
                )
            ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table in enumerate(self.block_tables):
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)

    def compute_slot_mapping_draft(
        self,
        req_indices: np.ndarray,
        positions: np.ndarray,
        positions_compressed_list: list[np.ndarray] | None = None,
        req_indices_compressed_list: list[np.ndarray] | None = None,
    ) -> None:
        for i, block_table in enumerate(self.block_tables):
            if positions_compressed_list and req_indices_compressed_list:
                block_table.compute_slot_mapping_draft(req_indices_compressed_list[i], positions_compressed_list[i])
            else:
                block_table.compute_slot_mapping_draft(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
