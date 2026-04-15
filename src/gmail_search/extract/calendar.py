"""Extract text from calendar invites (.ics / text/calendar)."""

import logging
from pathlib import Path
from typing import Any

from gmail_search.extract import ExtractResult

logger = logging.getLogger(__name__)


def extract_calendar(file_path: Path, config: dict[str, Any]) -> ExtractResult | None:
    """Parse iCal file and extract event summary, location, attendees, and description."""
    from icalendar import Calendar

    try:
        cal = Calendar.from_ical(file_path.read_bytes())
    except Exception as e:
        logger.warning(f"Failed to parse calendar file {file_path}: {e}")
        return None

    parts: list[str] = []

    for component in cal.walk():
        if component.name == "VEVENT":
            summary = str(component.get("SUMMARY", ""))
            location = str(component.get("LOCATION", ""))
            description = str(component.get("DESCRIPTION", ""))
            organizer = str(component.get("ORGANIZER", ""))
            dtstart = component.get("DTSTART")
            dtend = component.get("DTEND")

            lines = []
            if summary:
                lines.append(f"Event: {summary}")
            if location:
                lines.append(f"Location: {location}")
            if dtstart:
                lines.append(f"Start: {dtstart.dt}")
            if dtend:
                lines.append(f"End: {dtend.dt}")
            if organizer:
                lines.append(f"Organizer: {organizer}")

            attendees = component.get("ATTENDEE")
            if attendees:
                if isinstance(attendees, list):
                    lines.append(f"Attendees: {', '.join(str(a) for a in attendees[:10])}")
                else:
                    lines.append(f"Attendee: {attendees}")

            if description:
                lines.append(f"Description: {description[:500]}")

            parts.append("\n".join(lines))

    if not parts:
        return None

    return ExtractResult(text="\n\n".join(parts))
