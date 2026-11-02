from concurrent.futures import ThreadPoolExecutor

import pytest

from poker_decision.demo_limits import CallBudget, CallLimitError, CallLimits


def test_global_cap_is_shared_by_sessions_and_reopens(tmp_path):
    path = tmp_path / "usage.sqlite3"
    limits = CallLimits(daily=2, session_daily=2, cooldown_seconds=0)
    budget = CallBudget(path, limits)
    budget.reserve("a", now=100000)
    CallBudget(path, limits).reserve("b", now=100001)
    with pytest.raises(CallLimitError, match="shared"):
        budget.reserve("c", now=100002)
    budget.reserve("c", now=200000)


def test_session_limit_and_cooldown_do_not_consume_rejected_reservations(tmp_path):
    budget = CallBudget(tmp_path / "usage.sqlite3", CallLimits(10, 2, 5))
    budget.reserve("a", now=100000)
    with pytest.raises(CallLimitError, match="wait"):
        budget.reserve("a", now=100001)
    budget.reserve("a", now=100005)
    with pytest.raises(CallLimitError, match="session"):
        budget.reserve("a", now=100010)
    budget.reserve("b", now=100010)


def test_concurrent_requests_cannot_exceed_global_cap(tmp_path):
    budget = CallBudget(tmp_path / "usage.sqlite3", CallLimits(5, 5, 0))

    def attempt(i):
        try:
            budget.reserve(str(i), now=100000)
            return True
        except CallLimitError:
            return False

    with ThreadPoolExecutor(max_workers=10) as workers:
        assert sum(workers.map(attempt, range(20))) == 5
