"""生成工具静态安全检查。"""

from __future__ import annotations

from so_agent.runtime.tool_safety import assess_tool_source

SAFE_SOURCE = '''
import math
import json

def run(a, b):
    return {"sum": a + b, "sqrt": math.sqrt(abs(a))}
'''


def test_safe_pure_tool_passes():
    assessment = assess_tool_source(SAFE_SOURCE, tool_name="adder")
    assert assessment.passed is True
    assert assessment.issues == []


def test_network_import_rejected():
    source = "import requests\n\ndef run(url):\n    return requests.get(url).text\n"
    assessment = assess_tool_source(source, tool_name="fetcher")
    assert assessment.passed is False
    assert any("requests" in issue for issue in assessment.issues)


def test_subprocess_and_dynamic_execution_rejected():
    source = "import subprocess\n\ndef run(cmd):\n    return subprocess.check_output(cmd)\n"
    assert assess_tool_source(source).passed is False

    dynamic = "def run(x):\n    return eval(x)\n"
    assessment = assess_tool_source(dynamic)
    assert assessment.passed is False
    assert any("eval" in issue for issue in assessment.issues)


def test_dangerous_os_operations_rejected():
    source = "import os\n\ndef run(path):\n    os.remove(path)\n    return 1\n"
    assessment = assess_tool_source(source)
    assert assessment.passed is False
    assert any("os.remove" in issue for issue in assessment.issues)


def test_open_for_write_is_warning_not_rejection():
    source = "def run(path):\n    with open(path, 'w') as f:\n        f.write('x')\n    return 1\n"
    assessment = assess_tool_source(source)
    assert assessment.passed is True
    assert assessment.warnings


def test_syntax_error_rejected():
    assessment = assess_tool_source("def run(:\n    pass\n")
    assert assessment.passed is False


def test_oversized_source_rejected():
    assessment = assess_tool_source("#" * 200_001)
    assert assessment.passed is False


def test_sys_modules_escape_rejected():
    source = (
        "import sys\n\n"
        "def run():\n"
        "    return sys.modules['os'].system('echo bypass')\n"
    )
    assessment = assess_tool_source(source)
    assert assessment.passed is False
    assert any("sys.modules" in issue for issue in assessment.issues)

    from_import = "from sys import modules\n\ndef run():\n    return modules\n"
    assert assess_tool_source(from_import).passed is False


def test_introspection_builtins_rejected():
    for source in (
        "def run(x):\n    return getattr(x, 'secret')\n",
        "def run():\n    return vars(object())\n",
        "def run():\n    return globals()\n",
        "def run():\n    return locals()\n",
        "def run(x):\n    return x.__class__.__bases__\n",
        "def run(x):\n    return x.__dict__\n",
    ):
        assessment = assess_tool_source(source)
        assert assessment.passed is False, source


def test_import_aliases_cannot_bypass_module_attr_denylist():
    for source in (
        "from os import system\n\ndef run():\n    return system('echo x')\n",
        "import os as o\n\ndef run():\n    return o.system('echo x')\n",
        "import sys as s\n\ndef run():\n    return s.modules['os']\n",
        "from os import *\n\ndef run():\n    return system('echo x')\n",
    ):
        assessment = assess_tool_source(source)
        assert assessment.passed is False, source


def test_ordinary_os_path_usage_still_passes():
    source = (
        "import os\n\n"
        "def run(name):\n"
        "    return os.path.join('work', name)\n"
    )
    assessment = assess_tool_source(source)
    assert assessment.passed is True, assessment.issues
