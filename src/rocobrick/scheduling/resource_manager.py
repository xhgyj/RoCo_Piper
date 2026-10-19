"""Atomic task leases, distinct from trajectory reservations and controller owners.

No timeout or cancellation implicitly releases a lease. This in-memory component
is single-event-loop only; persistence and recovery are not implemented yet.
"""

from rocobrick.contracts.phases import ResourceRequirement


class ResourceManager:
    """Provide all-or-nothing claims and explicitly authorized releases."""

    def __init__(self) -> None:
        """Start with no owners."""
        self._owners: dict[ResourceRequirement, str] = {}

    def conflicts(
        self, owner: str, resources: tuple[ResourceRequirement, ...]
    ) -> tuple[ResourceRequirement, ...]:
        """Return requested resources already owned by somebody else."""
        return tuple(
            sorted(
                (r for r in resources if self._owners.get(r, owner) != owner),
                key=lambda r: r.resource_id,
            )
        )

    def claim(self, owner: str, resources: tuple[ResourceRequirement, ...]) -> bool:
        """Claim all resources or none; repeat claims are idempotent.

        Returns:
            Whether the complete request is owned by this caller.
        """
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("owner must be nonempty")
        if not isinstance(resources, tuple) or any(
            not isinstance(r, ResourceRequirement) for r in resources
        ):
            raise ValueError("resources must be a tuple of ResourceRequirement")
        if any(not r.exclusive for r in resources):
            raise ValueError("shared task leases are not implemented")
        if len(set(resources)) != len(resources):
            raise ValueError("duplicate resource")
        if self.conflicts(owner, resources):
            return False
        self._owners.update((r, owner) for r in resources)
        return True

    def held(self, owner: str) -> tuple[ResourceRequirement, ...]:
        """Return an immutable snapshot of an owner's leases."""
        return tuple(
            sorted(
                (r for r, current in self._owners.items() if current == owner),
                key=lambda r: r.resource_id,
            )
        )

    def release_authorized(self, owner: str) -> None:
        """Release after caller verifies terminal safety and obligation disposition.

        This primitive is deliberately not a cancellation or expiry API. The
        scheduling kernel owns verification; adapters must not call it directly.
        """
        for resource in self.held(owner):
            del self._owners[resource]
