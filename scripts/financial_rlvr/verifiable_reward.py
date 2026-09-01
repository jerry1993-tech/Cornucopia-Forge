"""
Verifiable Reward Function（五维可验证奖励函数）
为 GRPO 训练构建的「五维组合、正误分流」可验证奖励函数。

设计思路（整体概览）：
    本文件实现一个面向「可验证答案（指定格式）输出）」任务的奖励函数，
    用于强化学习（GRPO）训练中，对 Qwen3.5 系列模型生成的完整文本（<think>\n推理\n</think>\n\n答案）打分。

    核心奖励由 5 个维度加权求和而成：

        total = w1·R1 + w2·R2 + w3·R3 + w4·R4 + w5·R5
              = 0.60·准确率 + 0.15·格式 + 0.10·n-gram重复惩罚 + 0.10·余弦长度 + 0.05·截断惩罚

    各维度定义与归一区间：
        R1 准确率 Acc      ∈ [0,1]    主信号：答案完全正确 = 1，错误 = 0
        R2 格式 Format     ∈ {0,1}    校验 是否同时具备 <think> 标签与可抽取的选项字母
        R3 n-gram重复 Rep  ∈ [-1,0]   负向：3-gram 重复占比 r 的相反数，防复读 hack
        R4 余弦长度 Len    ∈ [-0.5, 1.0]    分正误差异化长度激励（核心）：正确[0.0,1.0]鼓励精简、错误[-0.5,0.0]鼓励探索
        R5 截断惩罚 Trunc  ∈ {-1,0}   负向：达到 max_tokens 被截断（无 EOS）则 -1

    最终 total 不做手动缩放，理论值域约 [-0.20, 0.85]（正项上限 0.85，重复/截断/错误短答案拉低到负值），
    GRPO 内部会自动做组内归一，因此无需把整体再缩放到 [-1, 1]。

    关键实现要点：
        1. 「正误分流」：R4 长度奖励先依据 R1 是否命中分流——正确样本用余弦曲线随长度下降
           （短→1.0、长→0.0，鼓励精简、不灌水拉长），错误样本随长度上升（短→-0.5、长→0.0，
           鼓励充分探索、不过早放弃）；
        2. 「答案抽取」：根据「抽取函数」判断是否成功；
        3. 「长度口径」：优先用框架传入的 response_token_ids（真实 token 长度），退化到 jieba 分词统计；
        4. 「截断检测」：优先用框架传入的 finish_reason == 'length' / is_truncated。

    最后把 VerifiableReward 注册进 swift 的 orms 字典，供命令行 --reward_funcs verifiable_reward 引用。
"""
import json
import math
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import jieba

# 兼容性导入：优先使用 ms-swift 提供的 ORM 基类和 orms 注册表。
# 若未安装 ms-swift，则退化为普通 object，保证本文件可独立 import 和单元测试。
try:
    from swift.rewards import ORM, orms
    HAS_SWIFT = True
except ImportError:
    HAS_SWIFT = False
    ORM = object
    orms = {}



@dataclass
class RewardConfig:
    """
    奖励函数配置（数据类，集中管理所有可调超参数）

    五个维度一个权重，默认权重之和为 1.0（对应 fangan 中的「基础版」权重）：

        accuracy_weight   —— 准确率权重（主信号，主导梯度）
        format_weight     —— 格式权重（约束 \boxed{} 输出格式）
        repetition_weight —— n-gram 重复惩罚权重（防复读 hack，软约束）
        length_weight     —— 余弦长度奖励权重（控制推理长度、正误差异化探索）
        truncation_weight —— 截断惩罚权重（轻惩罚，仅做截断负反馈）

    其余字段：
        repetition_n_grams / repetition_max_penalty —— 重复检测的 n-gram 大小 / 最大惩罚幅度
        cosine_min/max_len_value_wrong/correct —— R4 余弦长度奖励正误两档的最小/最大长度奖励值
        cosine_max_len         —— R4 生成文本的最大长度限制（余弦周期 T）
        use_token_length       —— True 优先用 response_token_ids 计长，否则用分词数
        truncation_penalty     —— 被截断时的惩罚值（默认 -1.0）
        require_solution       —— 是否强制要求提供标准答案 solution
        enable_component_logging —— 是否输出各分量统计日志
        log_every_n_calls      —— 每 N 次调用打印一次分量统计
    """
    accuracy_weight: float = 0.60
    format_weight: float = 0.15
    repetition_weight: float = 0.10
    length_weight: float = 0.10
    truncation_weight: float = 0.05

    # R3 重复检测（与 ms-swift 的 RepetitionPenalty 口径一致）
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0

    # R4 长度激励（余弦长度奖励，与 ms-swift CosineReward 口径一致）
    cosine_min_len_value_wrong: float = -0.5   # 错误答案：最小长度对应奖励
    cosine_max_len_value_wrong: float = 0.0    # 错误答案：最大长度对应奖励
    cosine_min_len_value_correct: float = 1.0  # 正确答案：最小长度对应奖励
    cosine_max_len_value_correct: float = 0.0  # 正确答案：最大长度对应奖励
    cosine_max_len: float = 9216               # 生成文本最大长度限制（余弦周期 T，默认值等于模型生成的最大程度）
    use_token_length: bool = True

    # R5 截断
    truncation_penalty: float = -1.0

    require_solution: bool = True
    enable_component_logging: bool = True
    log_every_n_calls: int = 20

    # 选项列表（单选题的合法选项字母）
    choices: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])



class VerifiableReward(ORM if HAS_SWIFT else object):
    """
    基于可验证答案的五维通用奖励函数（本文件核心类）

    继承自 swift 的 ORM（Outcome Reward Model）基类，因此可以被 ms-swift 的
    GRPO 训练框架直接识别并调用。其 `__call__` 签名符合 ms-swift 对奖励函数的约定：
        __call__(completions, solution=None, **kwargs) -> List[float]

    框架会以 `VerifiableReward(args=args)` 实例化本类（因此 __init__ 兼容接收 args），
    并在每次采样后以 `reward_func(completions, **kwargs)` 调用，其中：
        completions        —— 模型生成的完整文本列表（含 <think> 推理 + 答案）
        solution           —— 标准答案列表（Ground Truth，来自数据集的 solution 列）
        response_token_ids —— 每个样本生成部分的 token id 列表（用于计长）
        finish_reason      —— 结束原因（'stop' / 'length'）
        is_truncated       —— 是否因达到 max_tokens 被截断（finish_reason == 'length'）

    支持的评估维度（对应 5 个奖励分量）：
        1. Accuracy   —— 答案是否与标准答案一致                        → R1 ∈ [0,1]
        2. Format     —— 规定的格式是否合规                           → R2 ∈ {0,1}
        3. Repetition —— n-gram 重复占比的相反数（负向）               → R3 ∈ [-1,0]
        4. Length     —— 分正误差异化的余弦长度激励（核心）              → R4 ∈ [-0.5, 1.0]
        5. Truncation —— 是否被 max_tokens 截断（负向）               → R5 ∈ {-1,0}
    """

    def __init__(self, args: Any = None, config: RewardConfig = None, **kwargs):
        """
        初始化奖励函数

        Args:
            args: ms-swift 的 GRPOConfig（框架实例化时传入），本类只存引用、不读取其字段，
                  所有可调超参数统一由 config 管理。
            config: RewardConfig 配置对象，控制各维度权重与开关。
        """
        if HAS_SWIFT:
            super().__init__(args, **kwargs)
        else:
            self.args = args
        self.config = config or RewardConfig()
        # 调用计数器：用于周期性输出分量统计日志
        self._call_count = 0
        # 输入契约日志只打印一次（首调），避免刷屏
        self._input_contract_logged = False
        self.logger = logging.getLogger(__name__)

    def __call__(
        self,
        completions: List[str],
        solution: List[str] = None,
        **kwargs
    ) -> List[float]:
        """
        ms-swift 奖励函数的统一入口

        这是 GRPO 训练框架在每次采样后会调用的方法。框架传入一批模型生成的文本
        （completions），以及（可选）标准答案 solution。额外上下文（截断标志、token
        长度等）通过 kwargs 传入，字段名在不同 ms-swift 版本间有差异，故此处做兼容解析。

        Args:
            completions: 模型生成文本列表
            solution: 标准答案（Ground Truth）列表，可为 None
            **kwargs: 可能包含 'response_token_ids'（生成 token 长度）、'finish_reason'、
                      'is_truncated'、'solutions' / 'target' / 'answer'（答案备用字段）

        Returns:
            每个 completion 对应的总奖励分数列表，长度与 completions 一致
        """
        # ---- 1. 兼容解析「标准答案 solution」来源 ----
        solution_source = "positional"
        if solution is None:
            solution = kwargs.get('solutions')
            solution_source = "solutions_kwarg"
        if solution is None:
            solution = kwargs.get('target')
            solution_source = "target_kwarg"
        if solution is None:
            solution = kwargs.get('answer')
            solution_source = "answer_kwarg"

        # ---- 2. 归一化 batch：把标量/单元素输入扩展成与 completions 等长的列表 ----
        solution = self._normalize_batch(solution, len(completions), "solution")
        response_token_ids = self._normalize_batch(
            kwargs.get('response_token_ids'), len(completions), "response_token_ids")
        is_truncated = self._normalize_batch(kwargs.get('is_truncated'), len(completions), "is_truncated")
        finish_reason = self._normalize_batch(kwargs.get('finish_reason'), len(completions), "finish_reason")

        # ---- 3. 首次调用打印一次输入契约（类型、batch 形状），便于排查框架传参问题 ----
        if not self._input_contract_logged:
            self._log_input_contract(
                completions, solution, response_token_ids, finish_reason, is_truncated, kwargs,
            )
            self._input_contract_logged = True

        # ---- 4. 若配置要求必须有标准答案，但存在空答案，则直接报错（提示数据转换方式） ----
        if self.config.require_solution and any(not item for item in solution):
            raise ValueError(
                "Reward requires a non-empty 'solution' for every completion. "
                "Regenerate GRPO data so the assistant reference is stored in the 'solution' field."
            )

        # ---- 5. 逐条计算奖励，并收集各分量明细用于日志 ----
        rewards = []
        details_batch = []
        for comp, sol, ids, trunc, reason in zip(
                completions, solution, response_token_ids, is_truncated, finish_reason):
            details = self.score_with_details(comp, sol, ids, trunc, reason)
            rewards.append(details['total'])
            details_batch.append(details)

        # ---- 6. 周期性地输出各分量统计（均值、通过率等） ----
        self._call_count += 1
        if (
            self.config.enable_component_logging
            and (self._call_count == 1 or self._call_count % self.config.log_every_n_calls == 0)
        ):
            self._log_component_summary(details_batch)

        return rewards

    def score_with_details(
        self,
        completion: str,
        solution: Any,
        token_ids: Any = None,
        is_truncated: Any = None,
        finish_reason: Any = None,
    ) -> Dict[str, Any]:
        """
        计算单条样本的完整奖励，并返回所有分量明细（供测试与日志使用）

        这是整个奖励逻辑的核心：先依次计算五个分量，再按权重线性组合得到 total。
        关键控制流是「正误分流」：
            - 先算 R1 准确率，得到 is_correct；
            - R4 长度奖励依据 is_correct 选择「精简曲线」或「探索曲线」。

        Args:
            completion: 单条模型输出文本（<think>\n推理\n</think>\n\n答案）
            solution: 单条标准答案（可为 None）
            token_ids: 生成部分的 token id 列表（可为 None，退化为字数计长）
            is_truncated: 是否被截断的布尔标志（可为 None）
            finish_reason: 结束原因字符串（'stop' / 'length'，可为 None）

        Returns:
            字典，包含 accuracy/format/repetition/length/truncation/is_correct/
            is_truncated/completion_len/total 等字段
        """
        # ---- 分量 1（主信号）：准确率，答案是否与标准答案一致 ----
        accuracy = self._evaluate_accuracy(completion, solution)
        is_correct = accuracy >= 1.0

        # ---- 分量 2：格式是否合规 ----
        format_score = self._evaluate_format(completion)

        # ---- 分量 3（负向）：3-gram 重复惩罚 ----
        repetition = self._evaluate_repetition(completion)

        # ---- 分量 4（核心）：余弦长度奖励，正误分流 ----
        actual_len = self._completion_length(completion, token_ids)
        length = self._evaluate_length(actual_len, is_correct)

        # ---- 分量 5（负向）：截断惩罚 ----
        truncation = self._evaluate_truncation(is_truncated, finish_reason)

        # ---- 线性组合：五项加权求和（total 不做缩放，交给 GRPO 组内归一） ----
        total = (
            self.config.accuracy_weight * accuracy
            + self.config.format_weight * format_score
            + self.config.repetition_weight * repetition
            + self.config.length_weight * length
            + self.config.truncation_weight * truncation
        )

        return {
            'accuracy': accuracy,
            'format': format_score,
            'repetition': repetition,
            'length': length,
            'truncation': truncation,
            'is_correct': is_correct,
            'is_truncated': bool(is_truncated or finish_reason == 'length'),
            'completion_len': actual_len,
            'total': total,
        }

    # ==================== 分量实现 ====================

    def _evaluate_accuracy(self, completion: str, solution: Any) -> float:
        """
        R1 准确率（单选题）：答案完全正确 = 1，错误 = 0

        本奖励函数仅支持「单选题」题型：从 completion 与 solution 中各自抽取选项
        字母（A-D），忽略大小写与空白后判等；任一方无法抽出选项字母则判错。
        """
        if not solution:
            return 0.0

        pred = self._extract_choice(completion)
        gold = self._extract_choice(solution)

        # 任一答案抽取失败（无法定位选项字母）则判错
        if not pred or not gold:
            return 0.0

        return 1.0 if pred.upper() == gold.upper() else 0.0

    def _evaluate_format(self, completion: str) -> float:
        """
        R2 格式：校验「规定的格式」是否合规

        分两档：
            - 有 `<think>` 与 `</think>` 标签，且 _extract_choice(completion) is not None → 1.0
            - 其余情况（缺标签 / 抽不出选项字母）                                        → 0.0
        """
        has_thinking_tags = '<think>' in completion and '</think>' in completion
        if has_thinking_tags and self._extract_choice(completion) is not None:
            return 1.0
        return 0.0

    def _evaluate_repetition(self, completion: str) -> float:
        """
        R3 n-gram 重复惩罚：R3 = -r，其中 r ∈ [0,1] 为重复占比

        r = 1 - (不同 n-gram 数量 / 总 n-gram 数量)，与 ms-swift 的 RepetitionPenalty
        口径一致：把生成文本按空白切词后提取 n-gram（默认 3-gram），重复越多 r 越大，
        惩罚越接近 repetition_max_penalty（-1.0）。完全无重复时 r=0 → R3=0。
        """
        # 切词（jieba 分词，过滤空白 token）
        words = [w for w in jieba.lcut(completion) if w.strip()]

        # 词数不足 n-gram 大小时无法构成重复，不惩罚
        if len(words) < self.config.repetition_n_grams:
            return 0.0

        n = self.config.repetition_n_grams
        ngrams = set()
        total = 0
        for i in range(len(words) - n + 1):
            ngrams.add(tuple(words[i:i + n]))
            total += 1

        # 重复占比 r ∈ [0,1]
        repetition_ratio = 1.0 - len(ngrams) / total
        # R3 = r * max_penalty，默认 max_penalty = -1.0 → R3 ∈ [-1,0]
        return repetition_ratio * self.config.repetition_max_penalty

    def _evaluate_length(self, actual_len: float, is_correct: bool) -> float:
        """
        R4 余弦长度奖励：分正误差异化长度激励（核心）

        与 ms-swift CosineReward 口径一致，用 4 个「最小/最大长度奖励值」做正误分流：
            1）样本正确：短 → cosine_min_len_value_correct(1.0)，长 → cosine_max_len_value_correct(0.0)，
               奖励随长度递减，鼓励精简、不灌水拉长；
            2）样本错误：短 → cosine_min_len_value_wrong(-0.5)，长 → cosine_max_len_value_wrong(0.0)，
               奖励随长度递增，鼓励充分探索、不过早放弃。
        T 取 cosine_max_len（生成文本最大长度限制），t 为本次 rollout 输出 token 长度。
        """
        T = max(self.config.cosine_max_len, 1.0)
        if is_correct:
            # 正确样本：min/max 交换（min=0.0，max=1.0），短 → 高、长 → 低
            min_value = self.config.cosine_max_len_value_correct
            max_value = self.config.cosine_min_len_value_correct
        else:
            # 错误样本：min=0.0，max=-0.5，短 → 低、长 → 高
            min_value = self.config.cosine_max_len_value_wrong
            max_value = self.config.cosine_min_len_value_wrong
        return self._cosine_fn(actual_len, T, min_value, max_value)

    def _evaluate_truncation(self, is_truncated: Any, finish_reason: Any) -> float:
        """
        R5 截断惩罚：检测输出是否因达到 max_tokens 被截断（无 EOS 终止）

        优先用框架传入的 is_truncated / finish_reason == 'length' 判断，
        被截断时返回 truncation_penalty（默认 -1.0），正常结束返回 0.0。
        """
        truncated = bool(is_truncated) or finish_reason == 'length'
        if truncated:
            return self.config.truncation_penalty
        return 0.0

    # ==================== 辅助函数 ====================

    def _extract_choice(self, response: Any) -> Optional[str]:
        """
        从文本中抽取「单选题」选项字母（A-D）

        兼容多种常见写法：
            1) 字母包裹在 \\boxed{} 内，如 "\\boxed{A}" / "$\\boxed{A}$"；
            2) 文本以单个选项字母开头，如 "A"、"A。正确"；
            3) 中文模式，如「答案：A」「选A」「A 选项正确」「A 当选」等。
        抽取失败返回 None，由调用方判错（避免 assert 在异常输入下打断训练）。
        """
        choices = self.config.choices

        if response is None:
            return None
        response = str(response)

        # 1) 优先看最后一个 \boxed{...} 的内容是否为单个选项字母
        boxed = re.findall(r'\\boxed\{([^}]*)\}', response)
        if boxed:
            inner = boxed[-1].strip().upper()
            if len(inner) == 1 and inner in choices:
                return inner

        # 2) 去除强调符号后，若文本以单个选项字母开头则直接返回
        norm = re.sub(r'[*`]', '', response).strip().upper()
        if norm:
            first = norm[0]
            if first in choices:
                return first

        # 3) 中文「答案：A」「选A」等模式
        patterns = [
            (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
            (r'答案(是|为)选项 ?([ABCD])', 2),
            (r'故?选择?：? ?([ABCD])', 1),
            (r'([ABCD]) ?选?项(是|为)?正确', 1),
            (r'正确的?选项(是|为) ?([ABCD])', 2),
            (r'答案(应该)?(是|为)([ABCD])', 3),
            (r'选项 ?([ABCD]) ?(是|为)?正确', 1),
            (r'选择答案 ?([ABCD])', 1),
            (r'答案?：?([ABCD])', 1),
            (r'([ABCD])(选?项)?是?符合题意', 1),
            (r'答案选项：? ?([ABCD])', 1),
            (r'答案(选项)?为(.*?)([ABCD])', 3),
        ]

        for pattern, idx in patterns:
            m = re.search(pattern, norm, re.M)
            if m:
                answer = m.group(idx).strip().upper()
                if answer in choices:
                    return answer

        # 4) 「正确选项是 X」等补充模式
        supplementary = [
            (r'正确?答案(的)?选项(是|为)：?\s*([ABCD])', 3),
            (r'正确(的)?选项(是|为)：?\s*([ABCD])', 3),
        ]

        for pattern, idx in supplementary:
            m = re.search(pattern, norm, re.M)
            if m:
                answer = m.group(idx).strip().upper()
                if answer in choices:
                    return answer

        # 5) 「A...当选」「A...正确」：取最后一个匹配的选项字母
        for pattern in [r'([ABCD])(.*?)当选', r'([ABCD])(.*?)正确']:
            matches = re.findall(pattern, norm, re.M)
            if matches:
                last = matches[-1]
                answer = last[0] if isinstance(last, tuple) else last
                answer = answer.strip().upper()
                if answer in choices:
                    return answer

        return None

    def _completion_length(self, completion: str, token_ids: Any) -> float:
        """
        计算本次 rollout 输出的长度（L_actual）

        优先用框架传入的 response_token_ids（真实 token 长度）；不可用时退化到
        对 completion 文本按 jieba 切词计数。
        """
        if self.config.use_token_length and token_ids is not None:
            return float(self._token_count(token_ids))
        return float(len(jieba.lcut(completion)))

    @staticmethod
    def _token_count(token_ids: Any) -> int:
        """
        统计 token id 的数量

        兼容两种形态：
            - 单轮：扁平 list[int]             → 直接取长度
            - 多轮：嵌套 list[list[int]]（每轮一段）→ 各轮长度求和
        """
        if isinstance(token_ids, list):
            if token_ids and isinstance(token_ids[0], list):
                return sum(len(turn) for turn in token_ids)
            return len(token_ids)
        return 0

    @staticmethod
    def _cosine_fn(t: float, T: float, min_value: float, max_value: float) -> float:
        """
        ms-swift CosineReward.cosfn 同款余弦映射

        把生成长度 t 映射到 [min_value, max_value] 区间的余弦值：
            cosfn(t, T, min, max) = max - (max - min) * (1 - cos(t·π / T)) / 2
        其中 T 为最大长度限制（cosine_max_len）。t∈[0,T] 时余弦单调、平滑过渡。
        """
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2
    
    @staticmethod
    def _normalize_batch(value: Any, expected_size: int, field_name: str) -> List[Any]:
        """
        把标量/字典/列表统一归一化成与 batch 等长的列表

        处理三种情况：
            - None           → 用 [None]*N 占位（保持 zip 对齐）
            - 标量(str/dict) → 复制 N 份（同一份 solution 对每个样本都适用）
            - 列表           → 校验长度必须等于 N，否则报错
        """
        if value is None:
            return [None] * expected_size
        if isinstance(value, (str, dict)):
            return [value] * expected_size
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{field_name} must be a scalar, list, or tuple")
        if len(value) != expected_size:
            raise ValueError(
                f"{field_name} batch size {len(value)} does not match "
                f"completions batch size {expected_size}"
            )
        return list(value)

    # ==================== 日志 ====================

    def _log_input_contract(
        self,
        completions: Any,
        solution: Any,
        response_token_ids: Any,
        finish_reason: Any,
        is_truncated: Any,
        kwargs: Dict[str, Any],
    ) -> None:
        """
        打印输入契约日志（仅类型与 batch 形状，不泄露 prompt / 标签原文）

        目的：在训练启动时一次性确认 ms-swift 传入的各字段来源与形状是否符合预期，
        便于快速定位「solution / response_token_ids 没传进来」这类数据管道问题。
        """
        def summarize(value: Any) -> Dict[str, Any]:
            """把任意值归纳为 {type, size, first_item_type} 的摘要，不暴露内容"""
            summary = {"type": type(value).__name__}
            if isinstance(value, (list, tuple)):
                summary["size"] = len(value)
                summary["first_item_type"] = type(value[0]).__name__ if value else None
            return summary

        payload = {
            "event": "reward_input_contract",
            "completions": summarize(completions),
            "solution": summarize(solution),
            "response_token_ids": summarize(response_token_ids),
            "finish_reason": summarize(finish_reason),
            "is_truncated": summarize(is_truncated),
            "kwarg_keys": sorted(str(key) for key in kwargs),
        }
        self.logger.info("REWARD_INPUT_CONTRACT %s", json.dumps(payload, ensure_ascii=False))

    def _log_component_summary(self, details_batch: List[Dict[str, Any]]) -> None:
        """周期性地打印一批样本各奖励分量的统计摘要（均值、通过率）"""
        if not details_batch:
            return
        # 需要统计均值的数值型分量字段
        numeric_fields = ['total', 'accuracy', 'format', 'repetition', 'length', 'truncation']
        summary = {
            'event': 'reward_component_summary',
            'call': self._call_count,
            'batch_size': len(details_batch),
            # 准确率（正确样本占比）与截断率
            'accuracy_rate': sum(d['is_correct'] for d in details_batch) / len(details_batch),
            'truncation_rate': sum(d['is_truncated'] for d in details_batch) / len(details_batch),
        }
        for field_name in numeric_fields:
            summary[f'{field_name}_mean'] = round(
                sum(float(d[field_name]) for d in details_batch) / len(details_batch), 6
            )
        self.logger.info("REWARD_METRICS %s", json.dumps(summary, ensure_ascii=False))


# ============ 模块级注册 ============

# 若安装了 ms-swift，则把 VerifiableReward 注册进全局 orms 字典，
# 这样命令行可用 --reward_funcs verifiable_reward 直接引用。
if HAS_SWIFT:
    if isinstance(orms, dict):
        orms['verifiable_reward'] = VerifiableReward
        print("[verifiable_reward.py] Registered reward function: verifiable_reward")


if __name__ == "__main__":
    # ---- 自测入口：构造默认配置的奖励函数并用几条 mock 单选题样本打分 ----
    reward_func = VerifiableReward()

    mock_solution = "A"

    test_cases = [
        # 1) 正确 + 格式合规（有 think 标签 + 抽出选项 A）
        "<think>重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 逐项分析后，A 选项符合题意。</think>\n\\boxed{A}",
        # 2) 错误 + 格式合规（有 think 标签 + 抽出选项 B）
        "<think>我猜答案是 B。</think>\n答案是 B。",
        # 3) 错误 + 格式不合规（有 think 标签但抽不出选项）
        "<think>推理……</think>\n\\boxed{}",
        # 4) 错误 + 格式合规 + 长文本（重复循环 + 探索）
        "<think>重复 重复 重复 重复 重复 重复 重复 重复 重复 重复 " * 3 + "</think>\n\\boxed{D}",
    ]

    for i, comp in enumerate(test_cases):
        details = reward_func.score_with_details(comp, mock_solution, token_ids=list(range(len(jieba.lcut(comp)))))
        print(json.dumps({"case": i + 1, **details}, ensure_ascii=False))
