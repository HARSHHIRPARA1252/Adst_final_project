"""Read-only import helpers for the public eSportsAmaze BMSD pages.

The site currently renders match data in a Next.js server-component payload.
This module extracts that embedded, public JSON data without browser automation.
The importer is intentionally explicit: the Flask app fetches only when the user
presses its refresh button, and a failed refresh leaves the last local snapshot
untouched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import re
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd


BMSD_BASE_URL = "https://esportsamaze.com/bgmi/tournaments/battlegrounds-mobile-india-showdown-2026"
BMSD_MATCHES_URL = f"{BMSD_BASE_URL}/matches"
BMSD_STATS_URL = f"{BMSD_BASE_URL}/statistics"
BMSD_PROGRESSION_URL = f"{BMSD_BASE_URL}/progression"
BMSD_FORMAT_URL = f"{BMSD_BASE_URL}/format"
_NEXT_CHUNK = re.compile(r"self\.__next_f\.push\(\[1,(\"(?:\\\\.|[^\"\\\\])*\")\]\)", re.DOTALL)


def _decoded_server_payloads(page_html: str) -> str:
    chunks: list[str] = []
    # Next.js streams arbitrary Flight strings, so a regex that expects every
    # escape sequence to be a particular shape can silently miss whole pages.
    # Let Python's JSON decoder consume each quoted JavaScript string instead.
    marker = re.compile(r"self\.__next_f\.push\(\[1,")
    for match in marker.finditer(page_html):
        try:
            value, _ = json.JSONDecoder().raw_decode(page_html[match.end():])
            if isinstance(value, str):
                chunks.append(value)
        except (json.JSONDecodeError, TypeError):
            continue
    # Some Next.js versions inline the initial payload without a push call.
    # Include decoded script bodies as a fallback for those deployments.
    if not chunks and "stageGroups" in page_html:
        chunks.append(page_html.replace('\\"', '"'))
    return "\n".join(chunks)


def _extract_json_property(payload: str, property_name: str) -> Any:
    """Extract the first balanced JSON value belonging to a serialized key."""
    pattern = re.compile(r'"' + re.escape(property_name) + r'"\s*:\s*')
    for match in pattern.finditer(payload):
        start = match.end()
        if start >= len(payload) or payload[start] not in "[{":
            continue
        opening = payload[start]
        closing = "}" if opening == "{" else "]"
        stack = [closing]
        in_string = False
        escaped = False
        for index in range(start + 1, len(payload)):
            char = payload[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "]}":
                if not stack or stack[-1] != char:
                    break
                stack.pop()
                if not stack:
                    raw = payload[start:index + 1]
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"The source page did not contain readable {property_name} data.")


def parse_stage_groups(page_html: str) -> list[dict[str, Any]]:
    payload = _decoded_server_payloads(page_html)
    stage_groups = _extract_json_property(payload, "stageGroups")
    if not isinstance(stage_groups, list):
        raise ValueError("BMSD source returned an invalid stage list.")
    return stage_groups


def _first_present(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return value
    return None


def _match_group(match: dict[str, Any]) -> str:
    explicit = _first_present(match, "groupName", "group", "lobbyName")
    if explicit:
        return str(explicit).strip()
    label = str(match.get("format", ""))
    found = re.search(r"\((?:Group\s+)?([^()]+)\)\s*$", label, flags=re.IGNORECASE)
    return found.group(1).strip() if found else ""


def _iso_date(value: Any) -> str:
    if not value:
        return ""
    text = str(value).removeprefix("$D")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ""

def stage_groups_to_tables(
    stage_groups: list[dict[str, Any]],
    checked_at: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert source stageGroups to completed scorecards and fixture rows."""
    checked = checked_at or datetime.now(timezone.utc).date().isoformat()
    matches: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []

    for stage_group in stage_groups:
        if not isinstance(stage_group, dict):
            continue

        stage = str(
            _first_present(stage_group, "stageName", "name")
            or "Unknown stage"
        ).strip()

        for match in stage_group.get("matches", []) or []:
            if not isinstance(match, dict):
                continue

            match_id = _first_present(match, "matchNumber", "matchId")
            overall = _first_present(
                match, "overallMatchNumber", "overallMatch", "matchNumber"
            )
            source_match_id = str(
                _first_present(match, "id", "matchKey") or ""
            )
            status = str(match.get("status", "UNKNOWN")).upper()
            group = _match_group(match)
            map_name = str(
                _first_present(match, "mapName", "map") or ""
            ).strip()
            date = _iso_date(match.get("scheduledAt"))

            results = match.get("teamResults") or []
            participants = (
                results
                or _first_present(
                    match, "teams", "participants", "squads"
                )
                or []
            )

            base = {
                "date": date,
                "overall_match": overall,
                "match": match_id,
                "source_match_id": source_match_id,
                "stage": stage,
                "group": group,
                "map": map_name,
                "status": status,
                "scheduled_at": str(
                    match.get("scheduledAt", "")
                ).removeprefix("$D"),
                "source_url": BMSD_MATCHES_URL,
                "source_checked": checked,
            }

            if not participants:
                fixtures.append({
                    **base,
                    "team": "",
                    "team_tag": "",
                })
                continue

            for participant in participants:
                if not isinstance(participant, dict):
                    continue

                # Read result fields from the participant itself.
                # Some sources nest the scorecard under "result".
                result = participant
                nested_result = participant.get("result")
                if isinstance(nested_result, dict):
                    result = {**participant, **nested_result}

                team_obj = participant.get("team") or participant
                if not isinstance(team_obj, dict):
                    team_obj = {}

                team_name = _first_present(
                    team_obj,
                    "name",
                    "teamName",
                    "displayName",
                    "shortName",
                    "tag",
                )
                team_tag = _first_present(
                    team_obj,
                    "shortName",
                    "tag",
                    "shortCode",
                    "code",
                )

                team_name = str(team_name or team_tag or "").strip()
                tag = str(team_tag or "").strip()

                if not team_name:
                    continue

                fixture = {
                    **base,
                    "team": team_name,
                    "team_tag": tag,
                }
                fixtures.append(fixture)

                if status != "COMPLETED":
                    continue

                rank = result.get("rank")
                try:
                    rank_number = float(rank) if rank is not None else None
                except (TypeError, ValueError):
                    rank_number = None

                wwcd_value = result.get("wwcd")
                if wwcd_value is None:
                    wwcd_value = result.get("winnerWinnerChickenDinner")

                if isinstance(wwcd_value, str):
                    wwcd_flag = (
                        wwcd_value.strip().casefold()
                        in {"true", "1", "yes", "winner", "wwcd"}
                    )
                elif wwcd_value is None:
                    wwcd_flag = rank_number == 1
                else:
                    wwcd_flag = bool(wwcd_value)

                matches.append({
                    **base,
                    "team": team_name,
                    "rank": rank,
                    "wwcd": wwcd_flag,
                    "elim_points": result.get(
                        "elimsPoints", result.get("elimPoints")
                    ),
                    "place_points": result.get("placePoints"),
                    "total_points": result.get("totalPoints"),
                })

    return pd.DataFrame(matches), pd.DataFrame(fixtures)



class _ProgressionParser(HTMLParser):
    """Read the server-rendered route cards from the public progression page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.routes: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.section_depth = 0
        self.capture = ""
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "section" and self.current is None and str(attrs_dict.get("id", "")).startswith("route-"):
            self.current = {"route_id": attrs_dict.get("id", ""), "name": "", "advancement": "", "teams": [], "stages": []}
            self.section_depth = 1
        elif self.current is not None and tag == "section":
            self.section_depth += 1
        if self.current is None:
            return
        title = str(attrs_dict.get("title", "") or "").strip()
        if title and title not in self.current["stages"]:
            self.current["stages"].append(title)
        classes = str(attrs_dict.get("class", "")).split()
        if tag == "h3":
            self.capture, self.text = "name", []
        elif tag == "span" and "ed-chip" in classes:
            self.capture, self.text = "advancement", []
        elif tag == "h4":
            self.capture, self.text = "team", []

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.text.append(data)

    def handle_endtag(self, tag: str) -> None:
        matching = {"h3": "name", "span": "advancement", "h4": "team"}
        if self.current is not None and tag in matching and self.capture == matching[tag]:
            value = re.sub(r"\s+", " ", " ".join(self.text)).strip()
            if value:
                if self.capture == "team":
                    self.current["teams"].append(value)
                else:
                    self.current[self.capture] = value
            self.capture, self.text = "", []
        if self.current is not None and tag == "section":
            self.section_depth -= 1
            if self.section_depth == 0:
                self.current["teams"] = list(dict.fromkeys(self.current["teams"]))
                self.current["stages"] = list(dict.fromkeys(self.current["stages"]))
                self.routes.append(self.current)
                self.current = None


def parse_progression_routes(page_html: str) -> list[dict[str, Any]]:
    """Extract official qualification routes, advancement labels, and teams."""
    parser = _ProgressionParser()
    parser.feed(page_html)
    return [route for route in parser.routes if route["name"] and route["teams"]]


def fetch_bmsd_source(timeout: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the public match page and return completed scores plus fixtures."""
    request = Request(
        BMSD_MATCHES_URL,
        headers={"User-Agent": "BGMI-Tournament-Analytics/1.0 (public scorecard refresh)"},
    )
    with urlopen(request, timeout=timeout) as response:
        page_html = response.read().decode("utf-8", errors="replace")
    groups = parse_stage_groups(page_html)
    completed, fixtures = stage_groups_to_tables(groups)
    if completed.empty:
        raise ValueError("The BMSD source page did not include any completed match scorecards.")
    return completed, fixtures


def fetch_bmsd_progression(timeout: int = 20) -> list[dict[str, Any]]:
    """Fetch officially published BMSD advancement routes and qualified teams."""
    request = Request(
        BMSD_PROGRESSION_URL,
        headers={"User-Agent": "BGMI-Tournament-Analytics/1.0 (public progression refresh)"},
    )
    with urlopen(request, timeout=timeout) as response:
        page_html = response.read().decode("utf-8", errors="replace")
    routes = parse_progression_routes(page_html)
    if not routes:
        raise ValueError("The BMSD progression page did not include readable qualification routes.")
    return routes


def parse_stage_format(page_html: str, checked_at: str | None = None) -> pd.DataFrame:
    """Extract the public stage roster templates, totals, and advancement rules."""
    payload = _decoded_server_payloads(page_html)
    stages = _extract_json_property(payload, "stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("The BMSD format page did not include readable stage details.")

    checked = checked_at or datetime.now(timezone.utc).date().isoformat()
    rows: list[dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        groups = stage.get("groups") or {}
        rules = stage.get("rules") or []
        rows.append({
            "stage": str(_first_present(stage, "name", "stageName") or "Unknown stage").strip(),
            "stage_id": str(stage.get("id", "")),
            "sequence": stage.get("sequence"),
            "start_date": str(stage.get("startDate", "")),
            "end_date": str(stage.get("endDate", "")),
            "total_matches": stage.get("totalMatches"),
            "matches_per_day": stage.get("matchesPerDay"),
            "teams_count": stage.get("teamsCount"),
            "groups_division": str(stage.get("groupsDivision", "")),
            "format_type": str(stage.get("formatType", "")),
            "groups_json": json.dumps(groups, ensure_ascii=False),
            "rules_json": json.dumps(rules, ensure_ascii=False),
            "source_checked": checked,
            "source_url": BMSD_FORMAT_URL,
        })
    if not rows:
        raise ValueError("No readable BMSD stage format rows were found.")
    return pd.DataFrame(rows)


def fetch_bmsd_format(timeout: int = 20) -> pd.DataFrame:
    """Fetch the published stage structure and match counts for BMSD."""
    request = Request(
        BMSD_FORMAT_URL,
        headers={"User-Agent": "BGMI-Tournament-Analytics/1.0 (public stage format refresh)"},
    )
    with urlopen(request, timeout=timeout) as response:
        page_html = response.read().decode("utf-8", errors="replace")
    return parse_stage_format(page_html)


def fetch_bmsd_player_stats(timeout: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the public player performance table and its per-match eliminations."""
    request = Request(
        BMSD_STATS_URL,
        headers={"User-Agent": "BGMI-Tournament-Analytics/1.0 (public player statistics refresh)"},
    )
    with urlopen(request, timeout=timeout) as response:
        page_html = response.read().decode("utf-8", errors="replace")
    payload = _decoded_server_payloads(page_html)
    player_rows = _extract_json_property(payload, "playerRows")
    if not isinstance(player_rows, list) or not player_rows:
        raise ValueError("The BMSD statistics page did not include player performance records.")

    checked = datetime.now(timezone.utc).date().isoformat()
    summary: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for player in player_rows:
        if not isinstance(player, dict):
            continue
        summary.append({
            "player_id": player.get("playerId", ""),
            "player": str(_first_present(player, "ign", "playerName", "name") or "").strip(),
            "team": str(_first_present(player, "teamName", "team", "teamTag") or "").strip(),
            "team_tag": str(_first_present(player, "teamTag", "tag") or "").strip(),
            "matches_played": player.get("matchesPlayed"),
            "total_elims": player.get("totalElims"),
            "avg_elims": player.get("avgElims"),
            "max_elims": player.get("maxElims"),
            "zero_elims_matches": player.get("zeroElimsMatches"),
            "five_plus_elims_matches": player.get("fivePlusElimsMatches"),
            "reported_mvp_awards": player.get("totalMvps"),
            "total_damage": player.get("totalDamage"),
            "total_assists": player.get("totalAssists"),
            "total_survival_time": player.get("totalSurvivalTime"),
            "source_checked": checked,
            "source_url": BMSD_STATS_URL,
        })
        match_stats = player.get("matchStats") or {}
        if not isinstance(match_stats, dict):
            continue
        for match_id, match in match_stats.items():
            if not isinstance(match, dict):
                continue
            match_rows.append({
                "player_id": player.get("playerId", ""),
                "player": str(_first_present(player, "ign", "playerName", "name") or "").strip(),
                "team": str(_first_present(player, "teamName", "team", "teamTag") or "").strip(),
                "source_match_id": str(match_id),
                "stage": str(_first_present(match, "stageName", "stage") or "").strip(),
                "group": str(_first_present(match, "groupName", "group") or "").strip(),
                "day": _first_present(match, "day", "matchDay"),
                "map": str(_first_present(match, "mapName", "map") or "").strip(),
                "elims": match.get("playerElims"),
                "source_checked": checked,
            })
    if not summary:
        raise ValueError("No readable BMSD player records were found.")
    return pd.DataFrame(summary), pd.DataFrame(match_rows)
