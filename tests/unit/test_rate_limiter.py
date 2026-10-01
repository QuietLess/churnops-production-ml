from app.security import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_limit_then_blocks_with_retry_after():
    clock = FakeClock()
    limiter = RateLimiter(limit=3, window_seconds=60, clock=clock)
    assert [limiter.hit("a") for _ in range(3)] == [None, None, None]
    clock.now += 10
    assert limiter.hit("a") == 50  # first hit expires 60s after it happened


def test_window_slides_and_clients_are_independent():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=60, clock=clock)
    assert limiter.hit("a") is None
    assert limiter.hit("b") is None  # another client has its own budget
    assert limiter.hit("a") is not None
    clock.now += 60
    assert limiter.hit("a") is None
