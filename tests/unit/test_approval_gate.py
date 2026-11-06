from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.runtime.approvals import (
    APPROVED,
    DENIED,
    EXPIRED,
    ApprovalStore,
)
from so_agent.runtime.approval_gate import (
    ApprovalDenied,
    HighRiskApprovalGate,
    build_approval_gate,
)
from so_agent.web.app import create_app


def _settings(tmp_path: Path, **kwargs) -> Settings:
    values = {
        "high_risk_approval_enabled": True,
        "high_risk_approval_timeout_seconds": 2.0,
        "high_risk_approval_poll_seconds": 0.05,
        "approval_store_path": str(tmp_path / "approvals.db"),
    }
    values.update(kwargs)
    return Settings(_env_file=None, **values)


def test_store_request_is_idempotent(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals.db")
    first, created = store.request(
        idempotency_key="k1",
        task_id="t1",
        tool_name="run_in_sandbox",
        risk="network_socket",
        summary="s",
        expires_ts=9_999_999_999.0,
    )
    second, created_again = store.request(
        idempotency_key="k1",
        task_id="t1",
        tool_name="run_in_sandbox",
        risk="network_socket",
        summary="s",
        expires_ts=9_999_999_999.0,
    )
    assert created is True
    assert created_again is False
    assert first.id == second.id

    decided = store.decide(first.id, approved=True, decided_by="test")
    assert decided.status == APPROVED
    with pytest.raises(ValueError):
        store.decide(first.id, approved=False)


def test_store_wait_expires(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals.db")
    request, _ = store.request(
        idempotency_key="k2",
        task_id="t1",
        tool_name="run_in_sandbox",
        risk="network_socket",
        summary="s",
        expires_ts=0.0,
    )
    result = asyncio.run(
        store.wait(request.id, timeout=0.1, poll_seconds=0.02)
    )
    assert result.status == EXPIRED


def test_gate_disabled_does_not_create_store(tmp_path: Path) -> None:
    path = tmp_path / "approvals.db"
    gate = build_approval_gate(
        Settings(_env_file=None, approval_store_path=str(path))
    )
    asyncio.run(
        gate.authorize(
            "run_in_sandbox", {"code": "import socket; socket.socket()"}, task_id="t1"
        )
    )
    assert not path.exists()


def test_gate_assessment_and_approval(tmp_path: Path) -> None:
    gate = HighRiskApprovalGate(_settings(tmp_path))
    assert gate.assess("run_in_sandbox", {"code": "print(1)"}) is None
    assessment = gate.assess(
        "run_in_sandbox", {"code": "import socket\nsocket.socket()"}
    )
    assert assessment is not None
    assert "network_socket" in assessment.risk

    async def scenario() -> None:
        task = asyncio.create_task(
            gate.authorize(
                "run_in_sandbox",
                {"code": "import socket; socket.socket()", "task_id": "t1"},
                task_id="t1",
            )
        )
        pending = []
        for _ in range(100):
            pending = gate.store.list(status="pending")
            if pending:
                break
            await asyncio.sleep(0.02)
        assert pending
        gate.decide(pending[0].id, approved=True, decided_by="test")
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())


def test_gate_denial_and_reuse(tmp_path: Path) -> None:
    gate = HighRiskApprovalGate(_settings(tmp_path))
    arguments = {"code": "import subprocess", "task_id": "t1"}

    async def deny_once() -> str:
        async def decide_later() -> None:
            for _ in range(100):
                pending = gate.store.list(status="pending")
                if pending:
                    gate.decide(
                        pending[0].id,
                        approved=False,
                        decided_by="test",
                        reason="不允许",
                    )
                    return
                await asyncio.sleep(0.02)

        task = asyncio.create_task(
            gate.authorize("run_in_sandbox", arguments, task_id="t1")
        )
        decider = asyncio.create_task(decide_later())
        with pytest.raises(ApprovalDenied) as exc:
            await asyncio.wait_for(task, timeout=2)
        await decider
        return exc.value.request_id

    request_id = asyncio.run(deny_once())
    assert request_id
    first = gate.store.get(request_id)
    assert first is not None and first.status == DENIED

    # 同一任务/工具/参数重试：复用已拒绝记录，不重复创建。
    with pytest.raises(ApprovalDenied):
        asyncio.run(
            gate.authorize("run_in_sandbox", arguments, task_id="t1")
        )
    assert len(gate.store.list(status=DENIED)) == 1


def test_gate_timeout_denies(tmp_path: Path) -> None:
    gate = HighRiskApprovalGate(
        _settings(
            tmp_path,
            high_risk_approval_timeout_seconds=0.15,
            high_risk_approval_poll_seconds=0.02,
        )
    )
    with pytest.raises(ApprovalDenied):
        asyncio.run(
            gate.authorize(
                "run_in_sandbox", {"code": "import socket; socket.socket()"}, task_id="t1"
            )
        )
    expired = gate.store.list(status=EXPIRED)
    assert len(expired) == 1


def test_web_approval_endpoints(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "persistent" / "tool_approvals.db")
    request, _ = store.request(
        idempotency_key="web-key",
        task_id="t1",
        tool_name="run_in_sandbox",
        risk="network_socket",
        summary="命中网络访问",
        expires_ts=9_999_999_999.0,
    )
    app = create_app(
        settings=Settings(
            _env_file=None, approval_store_path="tool_approvals.db"
        ),
        base_dir=tmp_path,
    )
    with TestClient(app) as client:
        listed = client.get("/api/approvals").json()["approvals"]
        assert [item["approval_id"] for item in listed] == [request.id]

        response = client.post(
            f"/api/approvals/{request.id}",
            json={"approved": True, "reason": "测试"},
        )
        assert response.status_code == 200
        assert response.json()["approval"]["status"] == APPROVED

        conflict = client.post(
            f"/api/approvals/{request.id}", json={"approved": False}
        )
        assert conflict.status_code == 409
