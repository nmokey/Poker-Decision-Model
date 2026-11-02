"""Readable view of a Spot's context: card tiles, street, position, pot and the betting.

Parses the team's context format (see `build_inputs` in src/poker_decision/data.py):

    Street: turn. Hero position: IP. Hero cards: Kh 3h. Board: Th 3s 2d 5d.
    Preflop actions: BTN: raise 2.5bb; BB: call. Postflop actions: OOP: bet 4bb;
    IP: call; turn: 5d; OOP: check. Pot: 13 bb.

Display only. If the text doesn't match, the view falls back to the raw context, and
engines always receive the original string either way.
"""

import re
from dataclasses import dataclass

import streamlit as st

CARD = r"[2-9TJQKA][shdc]"
FIELDS = {
    "street": r"Street: (\w+)\.",
    "position": r"Hero position: (\w+)\.",
    "hero": rf"Hero cards: ({CARD} {CARD})\.",
    "board": rf"Board: ({CARD}(?: {CARD}){{2,4}})\.",
    "preflop": r"(?:Previous|Preflop) actions: (.*?)\. (?:Source|Postflop)",
    "postflop": r"Postflop actions: (.*?)\. Pot:",
    "pot": r"Pot: ([\d.]+) bb\.$",
}
SUITS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
POSITIONS = {"IP": "In position", "OOP": "Out of position"}
POSTFLOP_ORDER = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]  # first to act after the flop
SEATS = set(POSTFLOP_ORDER)
CSS = """<style>
.ps-row{display:flex;flex-wrap:wrap;gap:2.25rem;align-items:flex-end;margin:.25rem 0 1rem}
.ps-label{font-size:.75rem;letter-spacing:.04em;text-transform:uppercase;opacity:.65;margin-bottom:.35rem}
.ps-cards{display:flex;gap:.4rem}
.ps-card{width:2.7rem;height:3.7rem;border-radius:.35rem;background:#fff;border:1px solid #d0d7de;
display:flex;flex-direction:column;align-items:center;justify-content:center;
font-weight:600;font-size:1.05rem;line-height:1.15;color:#1f2328}
.ps-card.red{color:#c62828}
.ps-card.empty{background:transparent;border:1px dashed rgba(128,128,128,.45)}
.ps-value{font-size:1.15rem;font-weight:600;line-height:3.7rem;white-space:nowrap}
</style>"""


@dataclass(frozen=True)
class ParsedSpot:
    street: str
    position: str
    hero: list[str]
    board: list[str]
    betting: list[tuple[str, str]]  # (street name, what happened on it)
    pot: str


def _who(who: str, hero: str | None) -> str:
    return f"Hero ({who})" if who == hero else who


def _readable(history: str, hero: str | None) -> str:
    """'BTN: raise 2.5bb; BB: call' -> 'Hero (BTN) raise 2.5bb, BB call'."""
    if history == "none":
        return "no action yet"
    items = (item.partition(": ") for item in history.split("; "))
    return ", ".join(f"{_who(who, hero)} {what}" for who, _, what in items)


def _postflop_seats(preflop: str) -> dict[str, str]:
    """{'OOP': 'SB', 'IP': 'BB'}: the two seats in the preflop line, by postflop order.

    Postflop spots only say IP/OOP; this recovers the seat names for display.
    """
    seats = {item.partition(": ")[0] for item in preflop.split("; ")} & SEATS
    if len(seats) != 2:
        return {}
    oop, ip = sorted(seats, key=POSTFLOP_ORDER.index)
    return {"OOP": oop, "IP": ip}


def _postflop_betting(history: str, seats: dict[str, str], hero: str) -> list[tuple[str, str]]:
    """Split postflop history at the 'turn: Xx' / 'river: Xx' markers, one line per street."""
    if history == "none":
        return [("Flop", "no action yet")]
    lines, street, actions = [], "Flop", []
    for item in history.split("; "):
        who, _, what = item.partition(": ")
        if who in {"turn", "river"}:
            lines.append((street, ", ".join(actions) or "no action"))
            street, actions = who.capitalize(), []
        else:
            actions.append(f"{_who(seats.get(who, who), hero)} {what}")
    lines.append((street, ", ".join(actions) or "no action yet"))
    return lines


def parse_spot(context: str) -> ParsedSpot | None:
    """The fields of a context in the team's format, or None if it isn't in that format."""
    found = {name: re.search(pattern, context) for name, pattern in FIELDS.items()}
    if not (found["street"] and found["position"] and found["hero"] and found["pot"]):
        return None
    if found["preflop"] is None:
        return None
    street = found["street"][1]
    board = found["board"][1].split() if found["board"] else []
    if len(board) != {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(street, -1):
        return None
    position, preflop = found["position"][1], found["preflop"][1]
    seats = _postflop_seats(preflop) if position in POSITIONS else {}
    hero = seats.get(position, position)
    betting = [("Preflop", _readable(preflop, hero))]
    if street != "preflop":
        if found["postflop"] is None:
            return None
        betting += _postflop_betting(found["postflop"][1], seats, hero)
    if position in POSITIONS:
        side = POSITIONS[position]
        position = f"{hero}, {side.lower()}" if hero in SEATS else side  # "BB, in position"
    return ParsedSpot(
        street=street,
        position=position,
        hero=found["hero"][1].split(),
        board=board,
        betting=betting,
        pot=found["pot"][1],
    )


def card_html(card: str | None) -> str:
    if card is None:
        return '<div class="ps-card empty"></div>'
    rank = "10" if card[0] == "T" else card[0]
    red = " red" if card[1] in "hd" else ""
    return f'<div class="ps-card{red}"><span>{rank}</span><span>{SUITS[card[1]]}</span></div>'


def block(label: str, body: str) -> str:
    return f'<div><div class="ps-label">{label}</div>{body}</div>'


def draw_spot(context: str) -> None:
    spot = parse_spot(context)
    if spot is None:  # unknown format: show the text exactly as the engines get it
        st.markdown(context.replace("\n", "  \n"))
        return
    hero = "".join(card_html(c) for c in spot.hero)
    board = "".join(card_html(c) for c in spot.board + [None] * (5 - len(spot.board)))
    row = "".join(
        [
            block("Hero", f'<div class="ps-cards">{hero}</div>'),
            block("Board", f'<div class="ps-cards">{board}</div>'),
            block("Street", f'<div class="ps-value">{spot.street.capitalize()}</div>'),
            block("Position", f'<div class="ps-value">{spot.position}</div>'),
            block("Pot", f'<div class="ps-value">{spot.pot} bb</div>'),
        ]
    )
    st.markdown(f'{CSS}<div class="ps-row">{row}</div>', unsafe_allow_html=True)
    st.markdown("  \n".join(f"**{street}:** {what}" for street, what in spot.betting))
