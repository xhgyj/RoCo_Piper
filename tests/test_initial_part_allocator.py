from rocobrick.initial_part_allocator import allocate_initial_slots


def test_allocator_prefers_execution_robot_and_keeps_slots_separate():
    allocations = allocate_initial_slots(
        ("brick-b", "brick-a"),
        preferred_robots={"brick-a": "piper_0", "brick-b": "piper_0"},
        robot_base_xy={"piper_0": (0.24, -0.26), "piper_1": (-0.35, -0.30)},
        board_center_xy=(0.0, -0.05),
        board_half_extent_xy=(0.15, 0.15),
    )
    assert set(allocations) == {"brick-a", "brick-b"}
    assert all(item.preferred_robot_id == "piper_0" for item in allocations.values())
    assert allocations["brick-a"].position_xy_m != allocations["brick-b"].position_xy_m
    assert all(
        item.reason["allocator"] == "initial_part_allocator/v1"
        for item in allocations.values()
    )


def test_allocator_rejects_missing_preferred_robot():
    try:
        allocate_initial_slots(
            ("brick-a",),
            preferred_robots={"brick-a": "piper_2"},
            robot_base_xy={"piper_0": (0.24, -0.26)},
            board_center_xy=(0.0, -0.05),
            board_half_extent_xy=(0.15, 0.15),
        )
    except ValueError as error:
        assert "preferred robot" in str(error)
    else:
        raise AssertionError("missing preferred robot must fail")
