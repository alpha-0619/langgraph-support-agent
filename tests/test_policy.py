import pytest

from support_agent.policy import check


def test_clean_draft_passes():
    verdict = check("I have asked the finance team to look at the duplicate charge.")
    assert verdict.ok
    assert verdict.issues == []
    assert bool(verdict) is True


def test_empty_draft_fails():
    assert not check("   ")
    assert not check("")


@pytest.mark.parametrize(
    "draft",
    [
        "We guarantee this will not happen again.",
        "I promise you it is fixed.",
        "Our uptime is 100% secure.",
        "You will be refunded today.",
    ],
)
def test_unauthorised_commitments_are_caught(draft):
    verdict = check(draft)
    assert not verdict
    assert any("commitment" in issue for issue in verdict.issues)


@pytest.mark.parametrize(
    "draft,fragment",
    [
        ("Your refund of $250 is on its way.", "$250"),
        ("That takes 3 business days.", "3 business days"),
        ("We restored 40% of the data.", "40%"),
    ],
)
def test_numbers_absent_from_context_are_flagged(draft, fragment):
    verdict = check(draft, knowledge=["Refunds go back to the original payment method."])
    assert not verdict
    assert any(fragment in issue for issue in verdict.issues)


def test_number_present_in_retrieved_context_is_allowed():
    """The rule is about grounding, not about banning digits."""
    verdict = check(
        "Your refund of $250 will land on the original card.",
        knowledge=["Order 55-A was charged $250 twice on 2 August."],
    )
    assert verdict.ok, verdict.issues


def test_legal_advice_is_caught():
    verdict = check("You are not legally liable for that charge.")
    assert not verdict
    assert any("legal" in issue for issue in verdict.issues)


@pytest.mark.parametrize(
    "draft",
    [
        "I can see the card 4111 1111 1111 1111 on the account.",
        "I will follow up at jane.doe@example.com shortly.",
    ],
)
def test_pii_echo_is_caught(draft):
    verdict = check(draft)
    assert not verdict
    assert any("identifying" in issue for issue in verdict.issues)


def test_overlong_draft_is_caught():
    verdict = check("word " * 400)
    assert not verdict
    assert any("too long" in issue for issue in verdict.issues)


def test_all_broken_rules_are_reported_not_just_the_first():
    verdict = check("We guarantee a refund of $250 within 3 business days.")
    assert len(verdict.issues) >= 3


def test_issues_are_deduplicated():
    verdict = check("We guarantee $99. We guarantee $99.")
    assert len(verdict.issues) == len(set(verdict.issues))
