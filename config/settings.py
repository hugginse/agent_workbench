"""应用配置入口（骨架）。

完整 Settings 字段、.env 读取与校验逻辑在后续步骤实现；
业务代码禁止直接调用 os.getenv()，一律经本模块访问配置。
"""

from __future__ import annotations
