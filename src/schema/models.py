from typing import List

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """错误详情结构"""

    position: int = Field(description="错误在文本中的起始位置（字符索引）")
    # 为什么需要 position？
    # → 支持前端高亮显示，避免仅靠字符串匹配的歧义
    error_span: str = Field(description="错误的文本片段")
    error_type: str = Field(
        description="错误类型",
        examples=["拼写错误", "语法错误", "用词不当", "标点错误", "语义错误"],
    )
    # 为什么分类错误类型？
    # → 便于统计分析，后续可针对不同类型优化策略
    correction: str = Field(description="建议的修正")
    reason: str = Field(description="修正理由（结合上下文解释）")
    # 为什么需要理由？
    # → 建立用户信任，避免“黑盒修正”


class ChunkCorrection(BaseModel):
    """长文本单块纠错明细。"""

    index: int = Field(description="块序号，从 0 开始")
    chunk_text: str = Field(description="该块原始文本")
    corrected_text: str = Field(description="该块纠错后文本")
    errors: List[ErrorDetail] = Field(default_factory=list, description="该块检测到的错误")
    confidence: float = Field(
        description="该块置信度（0-1）",
        ge=0.0,
        le=1.0,
        default=1.0,
    )
    explanation: str = Field(default="", description="该块修正说明")


class CorrectionResult(BaseModel):
    """完整纠错响应结构"""

    original_text: str = Field(description="原始输入文本")
    corrected_text: str = Field(description="纠正后的完整文本")
    errors: List[ErrorDetail] = Field(description="检测到的错误列表")
    confidence: float = Field(
        description="整体置信度（0-1）",
        ge=0.0,
        le=1.0,
        default=1.0,
    )
    explanation: str = Field(description="整体修正说明")
    chunks: List[ChunkCorrection] = Field(
        default_factory=list,
        description="长文本分块明细；短文本或未分块时为空列表",
    )
