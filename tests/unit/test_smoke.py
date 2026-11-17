"""冒烟测试：验证 pytest 与 pythonpath 配置可用。"""


def test_package_importable() -> None:
    """src 布局下 agent_workbench 包应可导入。"""
    import agent_workbench

    assert agent_workbench.__version__
