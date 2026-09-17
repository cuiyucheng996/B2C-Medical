"""兼容旧入口：信封中间件已迁到 middleware.envelope。"""

from middleware.envelope import CompanionToolEnvelopeMiddleware

__all__ = ["CompanionToolEnvelopeMiddleware"]
