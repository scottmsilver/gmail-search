"""Live-API eval harness for the invitation crawl guard's Gemini Flash
classifier.

This is NOT a unit test — it makes REAL Gemini calls and is skipped
unless `GEMINI_API_KEY` is set. Run it directly to see precision/recall
and per-case verdicts:

    GEMINI_API_KEY=... python tests/eval_invite_guard.py

or under pytest (auto-skips without a key):

    pytest tests/eval_invite_guard.py -q -s

The corpus below is hand-authored to cover the safety-critical cases,
especially actionable invitations whose action link is an OPAQUE
tokenized URL (the denylist's blind spot). Each case carries
`should_skip` — the CORRECT gate decision given the safety asymmetry
(wrongly accepting an old invite is BAD; wrongly skipping a link only
misses indexed content).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest


@dataclass
class Case:
    name: str
    subject: str
    body: str
    from_addr: str
    should_skip: bool  # ground-truth correct gate decision
    note: str = ""


# ── Labeled corpus (≈22 cases) ─────────────────────────────────────────
CORPUS: list[Case] = [
    # ---- Actionable invites with OPAQUE tokenized URLs (key failure case) ----
    Case(
        "opaque_rsvp_birthday",
        "You're invited to Maya's 40th!",
        "Maya is turning 40 and wants you there. Tap below to let her know "
        "if you can make it.\n\nRespond: https://evt.partyvendor.com/r/aGVsbG8x9Q\n\n"
        "Saturday, July 12 at 7pm.",
        "Maya <celebrate@partyvendor.com>",
        True,
        "Opaque token; action is RSVP. Must skip.",
    ),
    Case(
        "opaque_accept_decline_dinner",
        "Dinner party — please respond",
        "We'd love to have you over for dinner next Friday. Click to accept "
        "or decline:\n\nhttps://rsvp.hosted.io/x/Z3Vlc3RfMTI3\n\nHope you can come!",
        "Sam <sam@hosted.io>",
        True,
        "Opaque token; accept/decline. Must skip.",
    ),
    Case(
        "opaque_confirm_attendance",
        "Confirm your attendance",
        "Please confirm you'll be attending the team offsite.\n\n" "Confirm here: https://go.offsite.app/t/YWJjZGVm",
        "Events <noreply@offsite.app>",
        True,
        "Opaque token; confirm attendance. Must skip.",
    ),
    Case(
        "opaque_vote_poll",
        "Vote for our next book club pick",
        "Cast your vote for next month's book:\n\nhttps://poll.bookclub.org/v/cG9sbDk5",
        "Book Club <hello@bookclub.org>",
        True,
        "Vote link via opaque token — non-idempotent. Must skip.",
    ),
    # ---- Google Calendar invites ----
    Case(
        "gcal_invite",
        "Invitation: Q3 Planning @ Wed Jun 18, 2026 10am",
        "You have been invited to the following event.\n\nQ3 Planning\n"
        "When: Wed Jun 18 2026 10am\n\nYes  No  Maybe\n"
        "https://calendar.google.com/calendar/event?action=RESPOND&eid=abc",
        "Google Calendar <calendar-notification@google.com>",
        True,
        "Classic Google Calendar RSVP. Must skip.",
    ),
    Case(
        "gcal_updated_event",
        "Updated invitation: Lunch @ Fri Jun 20",
        "This event has been updated.\nLunch\nFri Jun 20 12pm\n" "Going? Yes No Maybe",
        "Google Calendar <calendar-notification@google.com>",
        True,
        "Calendar response prompt. Must skip.",
    ),
    # ---- Evite / Paperless Post ----
    Case(
        "evite_party",
        "Tom invited you to a Backyard BBQ",
        "You're invited!\n\nBackyard BBQ\nSunday at 2pm\n\n"
        "Reply to this invitation: https://www.evite.com/event/00ABCDEF/reply",
        "Evite <no-reply@evite.com>",
        True,
        "Evite reply link. Must skip.",
    ),
    Case(
        "paperlesspost_wedding",
        "You're invited: Jordan & Alex's Wedding",
        "Please respond to the invitation.\n\nRSVP: https://www.paperlesspost.com/go/aB3xQ",
        "Paperless Post <invitations@paperlesspost.com>",
        True,
        "Paperless Post RSVP. Must skip.",
    ),
    # ---- 'Please confirm your email' account emails ----
    Case(
        "confirm_email_signup",
        "Confirm your email address",
        "Thanks for signing up. Please confirm your email address to "
        "activate your account:\n\nhttps://app.service.com/confirm/eyJhbGci",
        "Service <no-reply@service.com>",
        False,
        "Account email-confirmation, NOT an event invitation. The denylist "
        "handles /confirm/; the guard should NOT treat this as an invite. "
        "Crawling misses nothing valuable but it's not the invite class.",
    ),
    Case(
        "verify_account",
        "Verify your account",
        "Click to verify your account and start using our app:\n\n"
        "https://accounts.example.com/verify?token=opaque123",
        "Example <noreply@example.com>",
        False,
        "Account verification, not an event RSVP.",
    ),
    # ---- Benign event newsletters (mention an event, no personal action) ----
    Case(
        "event_newsletter_no_action",
        "This week: 5 concerts you shouldn't miss",
        "Summer is here and the city is buzzing with events. Here are our "
        "top picks for live music this week. Read the full guide: "
        "https://citymag.example.com/summer-concerts\n\n"
        "Tickets sold separately at each venue.",
        "City Magazine <newsletter@citymag.example.com>",
        False,
        "Newsletter mentions events but asks NO personal RSVP. Should crawl.",
    ),
    Case(
        "community_events_digest",
        "Your weekly community events digest",
        "Lots happening this week! Farmers market Saturday, library story "
        "time Tuesday, and a town hall Thursday. Full calendar: "
        "https://town.example.org/events",
        "Town News <digest@town.example.org>",
        False,
        "Digest of events, no action requested. Should crawl.",
    ),
    Case(
        "conference_announcement",
        "Registration open for DevConf 2026",
        "DevConf 2026 registration is now open. Learn more and grab your "
        "ticket: https://devconf.example.com/register\n\nEarly bird pricing "
        "ends soon.",
        "DevConf <hello@devconf.example.com>",
        False,
        "Promotional 'register' for a public conference, not a personal "
        "invitation RSVP. Should crawl (worst case: a registration page, "
        "not a silent accept).",
    ),
    # ---- Normal personal emails with article links ----
    Case(
        "personal_article_share",
        "Thought you'd like this",
        "Hey! Saw this article about sourdough and thought of you: "
        "https://cooking.example.com/sourdough-guide\n\nTalk soon!",
        "Dana <dana@gmail.com>",
        False,
        "Personal email sharing a link. Should crawl.",
    ),
    Case(
        "personal_catchup",
        "Catching up",
        "It's been ages! How are the kids? We should grab coffee sometime. "
        "Also here's that recipe: https://recipes.example.com/pasta",
        "Old Friend <friend@gmail.com>",
        False,
        "Personal, casual mention of meeting up — no formal invite/RSVP. " "Should crawl.",
    ),
    Case(
        "work_doc_link",
        "Notes from today's meeting",
        "Here are my notes from the standup: "
        "https://wiki.example.com/standup-2026-06-09\nLet me know if I "
        "missed anything.",
        "Coworker <coworker@work.example.com>",
        False,
        "Work doc link, no invitation. Should crawl.",
    ),
    # ---- Receipts ----
    Case(
        "receipt_order",
        "Your Acme order #12345",
        "Thanks for your order! Total: $42.99. Track your package: " "https://acme.example.com/track/12345",
        "Acme <orders@acme.example.com>",
        False,
        "Receipt with tracking link. Should crawl.",
    ),
    Case(
        "receipt_subscription",
        "Receipt for your subscription renewal",
        "Your annual subscription renewed. View your invoice: " "https://billing.example.com/invoice/abc",
        "Billing <billing@example.com>",
        False,
        "Billing receipt. Should crawl.",
    ),
    # ---- Edge: invite-shaped wording but informational ----
    Case(
        "invited_to_shop_promo",
        "You're invited to our VIP sale!",
        "You're invited to shop our exclusive VIP sale — 40% off "
        "everything. Shop now: https://shop.example.com/vip-sale",
        "Brand <promo@shop.example.com>",
        False,
        "Marketing 'you're invited' blast, not an event RSVP. Should crawl.",
    ),
    Case(
        "webinar_register_personal",
        "Please RSVP: Investor update call",
        "You're invited to our quarterly investor call. Please RSVP so we "
        "can send the dial-in:\n\nhttps://invites.fund.example/r/aW52ZXN0b3I",
        "Fund Relations <ir@fund.example>",
        True,
        "Personal RSVP request with opaque token. Must skip.",
    ),
    Case(
        "meeting_response_prompt",
        "Are you attending the all-hands?",
        "Quick check — are you attending Friday's all-hands? Let us know:\n" "https://hr.example.com/a/YWxsaGFuZHM",
        "HR <hr@example.com>",
        True,
        "'Are you attending' + opaque response link. Must skip.",
    ),
    Case(
        "plain_no_links",
        "lunch?",
        "wanna grab lunch tomorrow around noon? lmk",
        "Buddy <buddy@gmail.com>",
        False,
        "Casual, no links at all — nothing to skip. Should crawl (no-op).",
    ),
]


def _run_classifier(case: Case):
    from gmail_search.gmail import invite_guard

    verdict = invite_guard.classify_actionable_invitation(case.subject, case.body)
    # Mirror the gate: crawl ONLY on a confident benign verdict; any
    # unusable or unsure verdict counts as a skip (fail closed).
    skip = not (invite_guard._verdict_usable(verdict) and invite_guard._verdict_confidently_benign(verdict))
    return skip, verdict


def run_eval() -> dict:
    """Run the classifier over the corpus. Returns metrics + per-case rows."""
    rows = []
    tp = fp = tn = fn = 0
    for case in CORPUS:
        try:
            predicted_skip, verdict = _run_classifier(case)
        except Exception as e:  # fail-closed in production; here we record it
            predicted_skip, verdict = True, {"error": str(e)}
        correct = predicted_skip == case.should_skip
        if case.should_skip and predicted_skip:
            tp += 1
        elif case.should_skip and not predicted_skip:
            fn += 1  # MISSED an actionable invite — the dangerous error
        elif not case.should_skip and predicted_skip:
            fp += 1  # over-skip — only costs missed indexing
        else:
            tn += 1
        rows.append(
            {
                "name": case.name,
                "should_skip": case.should_skip,
                "predicted_skip": predicted_skip,
                "correct": correct,
                "verdict": verdict,
                "note": case.note,
            }
        )
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "rows": rows,
    }


def _print_report(metrics: dict) -> None:
    print("\n=== Invitation-guard classifier eval ===")
    print(f"recall (caught actionable invites): {metrics['recall']:.3f}  " f"(TP={metrics['tp']} FN={metrics['fn']})")
    print(
        f"precision (skip decisions correct): {metrics['precision']:.3f}  " f"(TP={metrics['tp']} FP={metrics['fp']})"
    )
    print(f"true-negatives (correctly crawled): {metrics['tn']}")
    print("\nPer-case:")
    for r in metrics["rows"]:
        flag = "OK " if r["correct"] else "XX "
        danger = "  <== DANGEROUS MISS" if (r["should_skip"] and not r["predicted_skip"]) else ""
        conf = ""
        if isinstance(r["verdict"], dict) and "confidence" in r["verdict"]:
            conf = f" conf={r['verdict'].get('confidence')}"
        print(
            f"  {flag}{r['name']:34s} want_skip={r['should_skip']!s:5s} "
            f"got={r['predicted_skip']!s:5s}{conf}{danger}"
        )


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs live GEMINI_API_KEY")
def test_eval_recall_and_precision():
    """Live eval gate: recall on truly-actionable invites must be ~1.0
    (fail-closed helps); precision should be reasonable."""
    metrics = run_eval()
    _print_report(metrics)
    # Recall is the safety-critical metric: never miss an actionable invite.
    assert metrics["recall"] >= 0.95, f"recall too low: {metrics['recall']:.3f} (FN={metrics['fn']})"
    # Precision floor — we tolerate some over-skip but not wholesale.
    assert metrics["precision"] >= 0.7, f"precision too low: {metrics['precision']:.3f} (FP={metrics['fp']})"


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY to run the live eval.")
    _print_report(run_eval())
