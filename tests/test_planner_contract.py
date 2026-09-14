from pathlib import Path
import re

from rocobrick.execution import Trajectory
from rocobrick.models import ActionType, Pose, Revisions, RobotSnapshot
from rocobrick.planner_client import _encode_request
from rocobrick.planner_contract import (
    CollisionObject,
    CollisionShape,
    PlanningRequest,
)


def test_cpp_and_python_protocols_are_identical():
    root = Path(__file__).parents[1]
    host = (root / "proto/planner/v1/planner.proto").read_text(encoding="utf-8")
    cpp = (
        root / "planner_ws/src/rocobrick_planner/proto/planner.proto"
    ).read_text(encoding="utf-8")
    assert re.sub(r"\s+", "", host) == re.sub(r"\s+", "", cpp)


def test_request_preserves_geometry_and_both_robot_states():
    revisions = Revisions(4, 5, 6)
    robots = (
        RobotSnapshot("piper_0", (0.0,) * 6, 1.0, "brick-a"),
        RobotSnapshot("piper_1", (0.1,) * 6, 1.0),
    )
    request = PlanningRequest(
        "request-1",
        ActionType.TRANSPORT,
        "piper_0",
        revisions,
        robots[0],
        Pose("world", (0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
        robots,
        (
            CollisionObject(
                "brick-a",
                Pose("world", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                CollisionShape.BOX,
                (0.032, 0.016, 0.0096),
                "sha256:test",
                "piper_0_tcp_link",
            ),
        ),
        Trajectory(tuple(f"piper_1_joint{i}" for i in range(1, 7)), ((0.1,) * 6,), (0.0,)),
        (),
    )

    encoded = _encode_request(request)

    assert tuple(encoded.objects[0].dimensions_m) == (0.032, 0.016, 0.0096)
    assert encoded.objects[0].attached_link == "piper_0_tcp_link"
    assert [robot.robot_id for robot in encoded.robots] == ["piper_0", "piper_1"]
    assert encoded.robots[0].holding_part == "brick-a"
