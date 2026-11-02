from dataclasses import dataclass


@dataclass(frozen=True)
class SpotDisplay:
    """UI-only visual details kept separate from the answer-free engine input."""

    street: str
    position: str
    pot: str
    hero_stack: str
    previous_actions: str
    hero_cards: tuple[str, ...]
    community_cards: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, cards in (
            ("hero_cards", self.hero_cards),
            ("community_cards", self.community_cards),
        ):
            if any(
                not isinstance(card, str)
                or len(card) != 2
                or card[0] not in "23456789TJQKA"
                or (card[1] not in "♠♥♦♣")
                for card in cards
            ):
                raise ValueError(f"{name} must contain rank/suit card codes")
        all_cards = self.hero_cards + self.community_cards
        if len(set(all_cards)) != len(all_cards):
            raise ValueError("A card cannot appear more than once in a spot")
        if len(self.hero_cards) != 2:
            raise ValueError("hero_cards must contain two cards")
        if len(self.community_cards) not in {0, 3, 4, 5}:
            raise ValueError("community_cards must contain 0, 3, 4, or 5 cards")
