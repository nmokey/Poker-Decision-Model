from __future__ import annotations

import html

from apps.display_details import SpotDisplay


def _render_cards(cards: tuple[str, ...], placeholder_count: int) -> str:
    if not cards:
        card_values: tuple[str | None, ...] = (None,) * placeholder_count
    elif len(cards) < placeholder_count:
        card_values = cards + (None,) * (placeholder_count - len(cards))
    else:
        card_values = cards

    rendered = []
    for card in card_values:
        if card is None:
            rendered.append('<div class="poker-card poker-card--back"></div>')
            continue
        rank, suit = card
        color = "red" if suit in "♥♦" else "black"
        rendered.append(
            '<div class="poker-card">'
            f'<span class="poker-card__corner poker-card__corner--{color}">'
            f"{html.escape(rank)}<br>{html.escape(suit)}</span>"
            f'<span class="poker-card__suit poker-card__suit--{color}">'
            f"{html.escape(suit)}</span></div>"
        )
    return '<div class="poker-card-row">' + "".join(rendered) + "</div>"


def _render_chips(label: str, amount: str) -> str:
    return (
        '<div class="poker-chip-stat">'
        '<div class="poker-chip-stack" aria-hidden="true">'
        '<span class="poker-chip poker-chip--blue"></span>'
        '<span class="poker-chip poker-chip--red"></span>'
        '<span class="poker-chip poker-chip--gold"></span>'
        "</div>"
        f'<span class="poker-chip-stat__label">{html.escape(label)}</span>'
        f'<strong class="poker-chip-stat__amount">{html.escape(amount)}</strong>'
        "</div>"
    )


def spot_visual_html(display: SpotDisplay) -> str:
    """Render structured spot details as a felt table and card rows."""

    hero_cards = _render_cards(display.hero_cards, 2)
    board_cards = _render_cards(display.community_cards, 5)
    return (
        '<section class="poker-table">'
        '<div class="poker-table__header">'
        f'<span class="poker-street">{html.escape(display.street.upper())}</span>'
        f'<span class="poker-position">HERO · {html.escape(display.position)}</span>'
        "</div>"
        '<div class="poker-felt">'
        f'<div class="poker-pot-display">{_render_chips("POT", display.pot)}</div>'
        '<div class="poker-board poker-board--center">'
        '<span class="poker-label">COMMUNITY CARDS</span>'
        f"{board_cards}"
        "</div>"
        '<div class="poker-player poker-player--center">'
        '<span class="poker-label">YOUR HAND</span>'
        f"{hero_cards}"
        f"{_render_chips('YOUR STACK', display.hero_stack) if display.hero_stack else ''}"
        "</div>"
        f'<div class="poker-action-history">{html.escape(display.previous_actions)}</div>'
        "</div>"
        "</section>"
    )
