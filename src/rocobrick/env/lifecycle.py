"""Isaac Kit lifecycle helpers for command-line workflows."""

from __future__ import annotations


async def close_kit_app(env, return_code: int) -> None:
    """Pause simulation and request an uncancellable Kit process shutdown."""
    if env is not None:
        try:
            await env.pause()
        except Exception as exc:  # noqa: BLE001 -- shutdown must still continue.
            print(f"[shutdown] WARNING: failed to pause simulation: {exc}")

    import omni.kit.app

    print(f"[shutdown] closing Isaac Kit (code={return_code})", flush=True)
    omni.kit.app.get_app().post_uncancellable_quit(return_code)
