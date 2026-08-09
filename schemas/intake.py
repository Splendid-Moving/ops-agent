"""
The shape of a job being booked.

Two models here, deliberately separate:

  ScreenshotExtraction — what the vision model produces. Every field is
    optional and carries a confidence plus the literal text it was read from,
    because a screenshot may show anything or nothing.

  JobIntake — the working record the graph carries and mutates as the user
    fills gaps. Plain values, no confidence.

Keeping them apart means the vision model can never write directly into the
record that drives real API calls. Everything crosses through validation first.
"""

from typing import Literal

from pydantic import BaseModel, Field

#: Below this, an extracted value is treated as missing and asked about.
#: Reading a phone number wrong is far more expensive than one extra question.
MIN_CONFIDENCE = 0.75


class ExtractedField(BaseModel):
    """One value read out of a screenshot, with its evidence."""

    value: str | None = Field(
        default=None,
        description="The extracted value, normalised. Null if not present in the image.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "How certain you are. 1.0 = printed clearly and unambiguously. "
            "Below 0.75 = the value will be double-checked with a human. "
            "Be honest; a confident wrong answer is the worst outcome."
        ),
    )
    source_text: str | None = Field(
        default=None,
        description=(
            "The literal text in the image this came from, copied verbatim. "
            "Lets a human audit the reading without reopening the screenshot."
        ),
    )

    @property
    def is_usable(self) -> bool:
        return bool(self.value) and self.confidence >= MIN_CONFIDENCE


class ScreenshotExtraction(BaseModel):
    """Everything the vision model could find in a customer screenshot."""

    full_name: ExtractedField = Field(default_factory=ExtractedField)
    email: ExtractedField = Field(default_factory=ExtractedField)
    phone: ExtractedField = Field(default_factory=ExtractedField)

    pickup_address: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Where the move starts. On a labor-only job, the only address.",
    )
    dropoff_address: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Where the move ends. Absent on labor-only jobs.",
    )
    extra_stop: ExtractedField = Field(
        default_factory=ExtractedField,
        description=(
            "A third address between pickup and drop-off, if and only if one is "
            "clearly shown. Never infer this."
        ),
    )

    # Occasionally present in a screenshot; never asked for if absent.
    move_date: ExtractedField = Field(default_factory=ExtractedField)
    arrival_time: ExtractedField = Field(default_factory=ExtractedField)
    movers: ExtractedField = Field(default_factory=ExtractedField)
    move_size: ExtractedField = Field(default_factory=ExtractedField)
    source: ExtractedField = Field(default_factory=ExtractedField)
    notes: ExtractedField = Field(default_factory=ExtractedField)

    is_labor: bool | None = Field(
        default=None,
        description=(
            "True if this is labor-only (loading/unloading help, no transport). "
            "False if it is a move between two addresses. Null if genuinely unclear. "
            "Do not guess from address count alone — that inference happens later "
            "in code. Set this only on explicit evidence, e.g. the words 'labor only', "
            "'loading help', 'just need help loading'."
        ),
    )

    overall_notes: str | None = Field(
        default=None,
        description=(
            "Anything else in the image a dispatcher would want to know: stairs, "
            "elevator, heavy items, parking constraints, timing constraints."
        ),
    )


class JobIntake(BaseModel):
    """
    The working record. Populated from extraction, then corrected and completed
    through conversation, then used to drive the four side effects.
    """

    full_name: str = ""
    email: str = ""
    phone: str = ""

    pickup_address: str = ""
    dropoff_address: str = ""
    extra_stop: str = ""

    move_date: str = ""       # mm/dd/yyyy
    arrival_time: str = ""    # "8-9am"
    movers: str = ""          # "2" | "3" | "4"

    is_labor: bool | None = None
    job_notes: str = ""
    #: True once the user has been asked about notes, even if they said none.
    notes_asked: bool = False

    move_size: str = ""
    source: str = ""

    #: Derived, not asked.
    rate: str = ""

    #: Per-address validation verdicts, keyed by field name.
    address_status: dict[str, str] = Field(default_factory=dict)

    def label(self) -> str:
        """Calendar event title. The '(labor)' suffix is load-bearing downstream."""
        name = self.full_name.strip() or "Unknown"
        return f"{name} (labor)" if self.is_labor else name


#: Fields the user can correct by name at the confirm gate.
EDITABLE_FIELDS: tuple[str, ...] = (
    "full_name", "email", "phone",
    "pickup_address", "dropoff_address", "extra_stop",
    "move_date", "arrival_time", "movers",
    "is_labor", "job_notes", "move_size", "source", "rate",
)

AddressField = Literal["pickup_address", "dropoff_address", "extra_stop"]
ADDRESS_FIELDS: tuple[str, ...] = ("pickup_address", "dropoff_address", "extra_stop")
