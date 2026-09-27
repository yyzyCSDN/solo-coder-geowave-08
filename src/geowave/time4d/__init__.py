"""四维地震交叉均衡（cross-equalization）流程。

阶段顺序固定为：覆盖匹配 -> 整体时移 -> 相位/振幅 -> 整形滤波器
-> 局部形变 -> 差异体与重复性图。所有估计结果与回放链保存在
``CrossRun`` 中；缺覆盖区域一律以 NaN + 可靠性码标记，绝不插成可靠变化。
"""
from geowave.time4d.config import CrossConfig
from geowave.time4d.types import Survey4D, CrossResult
from geowave.time4d.pipeline import run_cross_equalization
from geowave.time4d.summary import CrossSummary, summarize

__all__ = ['CrossConfig', 'Survey4D', 'CrossResult',
           'run_cross_equalization', 'CrossSummary', 'summarize']
