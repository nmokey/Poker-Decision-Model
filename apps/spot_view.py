import streamlit as st

from apps.context_view import parse_spot
from apps.display_details import SpotDisplay
from apps.felt_view import spot_visual_html
from apps.styles import CSS

SUITS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def draw_spot(context):
    parsed = parse_spot(context)
    if parsed is None:
        st.write(context)
        return

    def cards(values):
        return tuple(c[0] + SUITS[c[1]] for c in values)

    display = SpotDisplay(
        street=parsed.street.capitalize(),
        position=parsed.position,
        pot=parsed.pot + " bb",
        hero_stack="",
        previous_actions=" · ".join(f"{street}: {history}" for street, history in parsed.betting),
        hero_cards=cards(parsed.hero),
        community_cards=cards(parsed.board),
    )
    st.markdown(CSS + spot_visual_html(display), unsafe_allow_html=True)
