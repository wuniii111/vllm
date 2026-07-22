#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import copy  # 深拷贝，防止编译过程原地修改原始FX计算图
import functools  # 偏函数partial，固定参数递归调用
from collections.abc import Callable  # 类型注解：可调用函数
from typing import Any  # 通用任意类型注解

import torch
import torch.fx as fx  # PyTorch FX图捕获模块，用于算子图化简、编译
from torch._dynamo.backends.common import aot_autograd  # AOT自动求导，前后向编译隔离
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple  # FX图输出统一为tuple工具
from torch._inductor.decomposition import select_decomp_table  # 算子分解规则表（大算子拆小算子）
from torch.fx import GraphModule  # FX图模块载体
from vllm.compilation.compiler_interface import CompilerInterface  # vLLM统一编译器抽象基类
from vllm.config import VllmConfig  # vLLM全局总配置
from vllm.config.utils import Range  # 数值范围封装类，用于图捕获批次区间
from vllm.logger import logger  # vLLM日志工具

from vllm_ascend.ascend_config import AscendCompilationConfig, get_ascend_config  # 昇腾编译配置全局单例
from vllm_ascend.utils import COMPILATION_PASS_KEY  # 编译Pass管理器字典key常量


def compile_fx(graph: GraphModule, example_inputs: list, inner_compile: Callable, decompositions: dict) -> Callable:
    """
    封装AOT编译流程，统一处理FX图输出格式，接入内层自定义编译逻辑
    :param graph: 待编译的FX计算图
    :param example_inputs: 示例输入张量，用于推导shape
    :param inner_compile: 用户自定义内层编译回调
    :param decompositions: 算子分解规则表
    :return: 编译完成的可执行函数
    """
    # 生成递归偏函数，固定compile_fx入参，用于图输出转换时递归编译
    recursive_compile_fx = functools.partial(compile_fx, inner_compile=inner_compile, decompositions=decompositions)

    # 判断图返回值是否为tuple，torch.compile强制要求输出是元组
    if not graph_returns_tuple(graph):
        # 包装图，强制输出转为tuple，递归编译处理新图
        return make_graph_return_tuple(graph, example_inputs, recursive_compile_fx)
    # 走AOT Autograd编译链路，传入内层自定义编译回调
    return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)


def fusion_pass_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    compile_range: Range,
    key: str | None = None,
) -> tuple[Callable | None, Any | None]:
    """
    旧版编译分支：仅执行图融合Pass，不使用npugraph_ex/torchair ACL Graph，纯FX算子化简
    :param graph: FX计算图
    :param example_inputs: 示例输入
    :param compiler_config: 编译全局参数字典
    :param compile_range: 编译批次范围（未使用）
    :param key: 编译缓存key（未使用）
    :return: (编译后函数, 额外返回值None)
    """
    def compile_inner(graph, example_inputs):
        """内层编译回调：执行全部预定义图融合Pass"""
        # 取出注册好的算子融合/化简Pass管理器
        current_pass_manager = compiler_config[COMPILATION_PASS_KEY]
        # 对FX图执行一系列编译优化Pass（算子融合、常量折叠、无用节点删除等）
        graph = current_pass_manager(graph)
        return graph

    # 获取系统默认算子分解规则（大算子拆为基础算子）
    decompositions = select_decomp_table()

    # 调用上层通用FX编译封装函数，传入自定义内层编译逻辑
    compiled_fn = compile_fx(
        graph=graph,
        example_inputs=example_inputs,
        inner_compile=compile_inner,
        decompositions=decompositions,
    )

    # 返回编译完成的执行函数，无额外元数据
    return compiled_fn, None


##改
def _compute_decode_cudagraph_batch_sizes(vllm_config: VllmConfig) -> list[int]:
    # 1. 获取单次推测生成的draft候选token数量，无spec则为0
    num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
    # 2. 单条请求解码阶段单次query长度 = 真实1个token + spec候选token总数
    #    对应前面SFA代码里的 decode_threshold = num_spec_tokens + 1
    uniform_decode_query_len = num_spec_tokens + 1  # 上限受昇腾算子TND布局限制16
    # 3. 全局最大总token上限：最大并发请求数 × 单请求单次解码token数
    #    一批次最多同时存在的decode token总量
    max_num_tokens = vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len
    # 4. 遍历配置里预设的所有可捕获图尺寸，过滤满足两个条件的size：
    #    条件1：x >= uniform_decode_query_len 单请求最小token，不能小于单轮解码长度
    #    条件2：x <= max_num_tokens 不能超过全局批次token上限，否则显存/张量越界
    #    最终返回过滤后的合法batch size列表，用于Graph静态内核编译
    
    return [
        x
        for x in vllm_config.compilation_config.cudagraph_capture_sizes
        if max_num_tokens >= x >= 2
    ]

# def _compute_decode_cudagraph_batch_sizes(vllm_config: VllmConfig) -> list[int]:
#     # 1. 获取单次推测生成的draft候选token数量，无spec则为0
#     num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
#     # 2. 单条请求解码阶段单次query长度 = 真实1个token + spec候选token总数
#     #    对应前面SFA代码里的 decode_threshold = num_spec_tokens + 1
#     uniform_decode_query_len = num_spec_tokens + 1  # 上限受昇腾算子TND布局限制16
#     # 3. 全局最大总token上限：最大并发请求数 × 单请求单次解码token数
#     #    一批次最多同时存在的decode token总量
#     max_num_tokens = vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len
#     # 4. 遍历配置里预设的所有可捕获图尺寸，过滤满足两个条件的size：
#     #    条件1：x >= uniform_decode_query_len 单请求最小token，不能小于单轮解码长度
#     #    条件2：x <= max_num_tokens 不能超过全局批次token上限，否则显存/张量越界
#     #    最终返回过滤后的合法batch size列表，用于Graph静态内核编译
#     return [
#         x
#         for x in vllm_config.compilation_config.cudagraph_capture_sizes
#         if max_num_tokens >= x >= uniform_decode_query_len
#     ]


def _configure_backend(
    config: Any,  # 昇腾编译后端配置对象，分torchair / npugraph_ex两种后端
    ascend_compilation_config: AscendCompilationConfig,  # vLLM侧昇腾编译总开关配置
    vllm_config: VllmConfig,  # vLLM全局配置，包含spec、调度、图捕获尺寸等信息
    process_kwargs_options: Callable | None = None,  # 配置处理回调函数，区分新旧后端分支
) -> None:
    # 分支1：存在回调函数 = 使用新版 npugraph_ex 编译后端
    if process_kwargs_options is not None:
        # npugraph_ex 统一逻辑：构造参数字典，通过回调回填进后端config
        # 兼容新旧npugraph版本：旧版映射扁平参数到嵌套配置；新版直接赋值CompilerConfig
        # force_eager=True：图捕获前先用eager模式跑一遍FX图，做校验、shape推导
        # inplace_pass=False：关闭算子原地替换优化，防止Gelu算子降级到CPU引发拷数报错
        options: dict[str, Any] = {
            "force_eager": True,
            "inplace_pass": False,
        }
        # 如果开启静态Shape内核加速开关
        if ascend_compilation_config.enable_static_kernel:
            # 全局一次性打印日志，告知启用静态内核优化ACL Graph
            logger.info_once(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution.",
                scope="global",
            )
            # 开启静态内核编译总开关
            options["static_kernel_compile"] = True
            # 传入合法Graph batch尺寸列表，限定静态内核仅编译这些固定批次大小
            # 尺寸由前面 _compute_decode_cudagraph_batch_sizes 过滤spec适配的合法size
            options["_vllm_aclnn_static_kernel_sym_range"] = _compute_decode_cudagraph_batch_sizes(vllm_config)
        # 调用回调，将options字典写入后端编译配置
        process_kwargs_options(config, {"options": options})
    # 分支2：无回调函数 = 使用旧版 torchair reduce-overhead 后端
    else:
        # torchair 直接操作嵌套式config结构体，不通过扁平options字典
        # mode="reduce-overhead"：启用ACL Graph模式，跳过FX图转昇腾IR的冗余转换，降低开销
        config.mode = "reduce-overhead"
        # 开启预执行eager模式，捕获图前先跑一遍算子校验
        config.debug.run_eagerly = True
        # 关闭原地算子替换Pass，避免Gelu切CPU产生主机设备拷贝异常
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        # 同样判断是否开启静态Shape内核
        if ascend_compilation_config.enable_static_kernel:
            logger.info_once(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution.",
                scope="global",
            )
            # 开启ACL Graph静态Shape内核总开关
            config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
            # 把过滤后的合法Graph批次尺寸赋值给静态符号取值范围
            config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range = (
                _compute_decode_cudagraph_batch_sizes(vllm_config)
            )


def npugraph_ex_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    vllm_config: VllmConfig,
    ascend_compilation_config: AscendCompilationConfig,
    compile_range: Range,
    key: str | None = None,
) -> tuple[Callable | None, Any | None]:
    """
    新版ACL Graph编译入口：优先npugraph_ex，导入失败自动降级旧版torchair
    :param graph: FX计算图
    :param example_inputs: 示例输入
    :param compiler_config: 编译参数字典
    :param vllm_config: vLLM全局配置
    :param ascend_compilation_config: 昇腾专属编译开关
    :param compile_range: 编译批次范围（未使用）
    :param key: 编译缓存key（未使用）
    :return: (编译完成的Graph执行函数, None)
    """
    # 优先尝试新版npugraph_ex，导入失败回退torchair保证兼容性
    try:
        # 导入新版昇腾图编译SDK
        import npugraph_ex as nge

        # 关闭原生Torch NPU JIT，统一交由npugraph_ex接管编译
        torch.npu.set_compile_mode(jit_compile=False)
        # 实例化npugraph_ex编译器配置对象
        config = nge.CompilerConfig()
        # _process_kwargs_options存在新旧两个模块路径，做兼容导入
        try:
            # 新版npugraph_ex路径
            from npugraph_ex.configs.compiler_config import _process_kwargs_options
        except ImportError:
            # 旧版npugraph_ex路径兜底
            from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
        # 统一配置后端编译参数（静态内核、eager预跑、spec合法batch范围等）
        _configure_backend(
            config, ascend_compilation_config, vllm_config, process_kwargs_options=_process_kwargs_options
        )
        # 根据配置创建NPU Graph编译后端
        npugraph_ex = nge.get_npu_backend(compiler_config=config)
    except ImportError:
        # npugraph_ex不存在，降级到传统torchair编译链路
        import torchair

        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        # 无回调函数，走torchair嵌套配置分支
        _configure_backend(config, ascend_compilation_config, vllm_config)
        npugraph_ex = torchair.get_npu_backend(compiler_config=config)

    # torch.compile强制要求FX图输出为tuple，不满足则包装转换后再编译
    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, npugraph_ex), None
    # 调用昇腾后端编译FX图，返回可重放的ACL Graph执行函数
    return npugraph_ex(graph, example_inputs), None


class AscendCompiler(CompilerInterface):
    """
    vLLM昇腾平台自定义编译器实现类，遵循vLLM统一CompilerInterface抽象接口
    负责捕获PyTorch FX算子图，根据开关选择两种编译链路：
    1. enable_npugraph_ex=True：npugraph_ex/torchair ACL Graph静态图编译（高性能解码）
    2. enable_npugraph_ex=False：仅算子融合Pass，无ACL Graph加速
    """

    # 编译器唯一标识名称，vLLM调度区分不同硬件编译器
    name = "AscendCompiler"

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """
        计算编译缓存哈希，相同配置复用编译产物，避免重复编译
        :param vllm_config: vLLM全局配置
        :return: 配置哈希字符串
        """
        # 如果开启npugraph_ex，缓存绑定当前vllm全局配置（spec、图捕获尺寸会影响编译产物）
        npugraph_ex_enabled = get_ascend_config().ascend_compilation_config.enable_npugraph_ex
        if npugraph_ex_enabled:
            self.vllm_config = vllm_config
        # 复用vLLM内置配置哈希计算逻辑
        return vllm_config.compute_hash()

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:
        """
        vLLM编译器标准入口方法，每次图捕获都会调用
        :param graph: 原始FX算子计算图
        :param example_inputs: 示例输入张量，用于shape推导
        :param compiler_config: 编译全局参数字典，包含算子融合Pass
        :param compile_range: 编译批次范围
        :param key: 编译缓存标识key
        :return: (编译完成的可执行函数, 额外元数据None)
        """
        # inductor编译会原地修改原始图，深拷贝一份隔离原始图，规避bug
        # 参考PyTorch官方issue #138980
        graph = copy.deepcopy(graph)

        from torch._guards import detect_fake_mode
        # 检测当前是否处于FakeTensor虚拟shape推导模式
        current_fake_mode = detect_fake_mode()
        if current_fake_mode is not None:
            # 统一转换示例输入到当前FakeMode，保证shape推导一致
            example_inputs = [
                current_fake_mode.from_tensor(inp)
                if (
                    isinstance(inp, torch.Tensor)
                    and hasattr(inp, "fake_mode")
                    and inp.fake_mode is not current_fake_mode
                )
                else inp
                for inp in example_inputs
            ]

        # 读取全局单例中的昇腾编译配置
        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        # 判断是否开启高性能ACL Graph编译
        if ascend_compilation_config.enable_npugraph_ex:
            logger.info("enable_npugraph_ex is enabled, which will bring graph compilation optimization.")
            # 开启npugraph_ex时必须提前缓存vllm_config，供下游读取spec、图捕获尺寸
            assert hasattr(self, "vllm_config")
            # 走ACL Graph编译链路
            return npugraph_ex_compile(
                graph, example_inputs, compiler_config, self.vllm_config, ascend_compilation_config, compile_range, key
            )
        else:
            # 关闭npugraph_ex，仅做算子融合，不生成静态ACL Graph
            return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)