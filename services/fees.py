"""
Office fees read out of the job notes: gas fee, heavy item fee, discount.

The dispatcher already types these into the notes ("Gas fee $50", "piano $150").
The crew app has a column for each, and the bill is built from those columns,
so a fee that only lives in the notes never reaches the bill. This pulls the
dollar amounts out so the app gets them as numbers.

Deliberately strict, because a wrong fee is worse than a missing one:
  - Only amounts written with "$" count ("$50", "50$"). "3rd floor" or
    "2 trucks" can never become a fee.
  - Each fee must sit in its own piece of the note. "$150 heavy fee and $70
    gas fee" is split on "and"; a line that still mixes two kinds after that
    ("gas and piano $200") is skipped and reported, and the office fills it in.
  - The crew app stores the discount as a PERCENT (discount_percent). "10% off"
    / "10% discount" is sent as 10. A dollar discount ("$20 off") has no column
    to go in, so it is reported, not converted.

The confirm summary shows what was read, so the dispatcher sees the numbers
before anything is created.
"""

import re
from dataclasses import dataclass, field

_KINDS: dict[str, re.Pattern] = {
    "gas_fee": re.compile(r"\b(gas|fuel|travel)\b", re.IGNORECASE),
    "heavy_item_fee": re.compile(
        r"\b(heavy|piano|pianos|safe|safes|pool\s*table|hot\s*tub|treadmill|"
        r"gun\s*safe|aquarium|marble|organ)\b",
        re.IGNORECASE,
    ),
    "discount": re.compile(r"\b(discount|off)\b", re.IGNORECASE),
}

LABELS = {"gas_fee": "Gas", "heavy_item_fee": "Heavy item", "discount_percent": "Discount"}

# "$50", "$ 50", "$1,200.50", "50$"
_MONEY = re.compile(r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)|(\d[\d,]*(?:\.\d{1,2})?)\s*\$")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# A clause ends at a newline, ";", or ", " — never at the "." inside "$50.00".
_CLAUSE_SPLIT = re.compile(r"\n|;|,\s+|\.\s+|\s+\|\s+")
# Inside a clause naming two kinds of fee: "$150 heavy fee and $70 gas fee".
_AND_SPLIT = re.compile(r"\s+(?:and|plus|&|\+)\s+", re.IGNORECASE)


@dataclass
class Fees:
    gas_fee: float | None = None
    heavy_item_fee: float | None = None
    discount_percent: float | None = None
    #: Note text that mentions a fee but could not be read safely.
    unclear: list[str] = field(default_factory=list)

    def any(self) -> bool:
        return any(v for v in (self.gas_fee, self.heavy_item_fee, self.discount_percent))

    def summary(self) -> str:
        """'Gas $50 · Heavy item $150' — empty if nothing was read."""
        parts = []
        for key in ("gas_fee", "heavy_item_fee", "discount_percent"):
            value = getattr(self, key)
            if value:
                shown = f"{value:g}%" if key == "discount_percent" else _money(value)
                parts.append(f"{LABELS[key]} {shown}")
        return " · ".join(parts)


def _money(value: float) -> str:
    return f"${value:,.0f}" if value == int(value) else f"${value:,.2f}"


def _amounts(text: str) -> list[float]:
    out = []
    for m in _MONEY.finditer(text):
        raw = (m.group(1) or m.group(2)).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def _kinds(text: str) -> list[str]:
    kinds = [k for k, rx in _KINDS.items() if rx.search(text)]
    # "off" alone is too common ("drop off", "off-street") to count as a
    # discount unless a money amount or percent sits next to it.
    if "discount" in kinds and not re.search(r"\bdiscount\b", text, re.IGNORECASE):
        if not re.search(r"(\$\s*\d[\d,.]*|\d[\d,.]*\s*\$|\d+\s*%)\s*off\b", text, re.IGNORECASE):
            kinds.remove("discount")
    return kinds


def _clauses(notes: str) -> list[str]:
    """
    Note text cut into pieces that each name at most one kind of fee, where
    that can be done safely. A piece naming two kinds is split on "and" /
    "plus" / "&" — but only if every resulting part then names exactly one
    kind and carries its own amount; otherwise it stays whole and is reported
    as unclear.
    """
    out = []
    for clause in _CLAUSE_SPLIT.split(str(notes or "")):
        text = clause.strip().lstrip("-•*").strip()
        if not text:
            continue
        if len(_kinds(text)) > 1:
            parts = [p.strip() for p in _AND_SPLIT.split(text) if p.strip()]
            # Every part needs its own kind AND its own number, so
            # "gas and piano $200" can never hand all $200 to the piano.
            if len(parts) > 1 and all(
                len(_kinds(p)) == 1 and (_amounts(p) or _PERCENT.search(p)) for p in parts
            ):
                out.extend(parts)
                continue
        out.append(text)
    return out


def parse_fees(notes: str) -> Fees:
    fees = Fees()
    totals: dict[str, float] = {}

    for text in _clauses(notes):
        if not text:
            continue
        kinds = _kinds(text)
        if not kinds:
            continue

        amounts = _amounts(text)
        if len(kinds) > 1:
            if amounts or _PERCENT.search(text):
                fees.unclear.append(text)
            continue
        kind = kinds[0]

        if kind == "discount":
            percents = _PERCENT.findall(text)
            if len(percents) == 1 and not amounts and 0 < float(percents[0]) < 100:
                fees.discount_percent = float(percents[0])
            elif percents or amounts:
                fees.unclear.append(text)   # "$20 off", or two numbers
            continue

        if amounts:
            totals[kind] = totals.get(kind, 0.0) + sum(amounts)

    for kind, total in totals.items():
        if total > 0:
            setattr(fees, kind, round(total, 2))
    return fees
