"""Common dependencies for FastAPI routes."""

from fastapi_throttle import RateLimiter

# Rate limiter instance for routes
router_limiter = RateLimiter(times=5, seconds=30)

