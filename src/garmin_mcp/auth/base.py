"""AuthProvider interface — the single seam between HTTP auth and Garmin client resolution.

Phase 1: GarminTokenAuthProvider (bearer token = base64 Garmin OAuth token).
Phase 2 (multi-user): implement a new subclass that looks up tokens per-user in a store,
then pass it to GarminAuthMiddleware — no other code needs to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class UserContext:
    """Resolved identity for a single authenticated request."""

    user_id: str
    garmin_client: object  # raw Garmin client; middleware wraps in _GarminProxy before configuring modules


class AuthProvider(ABC):
    @abstractmethod
    def resolve(self, token: str) -> UserContext | None:
        """Return a UserContext if the token is valid, None to reject the request.

        Implementations must be safe to call concurrently (asyncio event loop).
        Returning None causes the middleware to respond with HTTP 401.
        """
        ...
