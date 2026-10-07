"""Session-only robot visual instance expansion; no physics/model mutation."""


def materialize_robot_visuals(stage: object, robot_path: str) -> tuple[str, ...]:
    """Expand visual references under their moving body, not shared prototypes.

    All candidate subtrees are checked before changing instance metadata.
    Rigid bodies/colliders are forbidden. Edits belong to the session layer,
    so original USD assets and their physics/collision instancing stay intact.

    Returns:
        The precise visual roots materialized for this runtime session.
    """
    from pxr import Usd, UsdPhysics

    robot = stage.GetPrimAtPath(robot_path)
    if not robot.IsValid():
        raise ValueError("visual repair requires a loaded robot root")
    roots = []
    for body in robot.GetChildren():
        visual = body.GetChild("visuals")
        if not visual.IsValid() or not visual.IsInstance():
            continue
        for prim in Usd.PrimRange(visual, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                raise ValueError(f"visual subtree contains physics: {prim.GetPath()}")
        roots.append(visual)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for visual in roots:
            visual.SetInstanceable(False)
    return tuple(str(visual.GetPath()) for visual in roots)
