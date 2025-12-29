from pydantic import BaseModel, NonNegativeInt, Field

from data_structuring.components.tags import Tag, BIOTag


class TaggedSpan(BaseModel, frozen=True):
    """A span within a transaction and its corresponding tag."""

    start: NonNegativeInt = Field(description="The absolute index of the first letter of the match within the message")
    end: NonNegativeInt = Field(description="The absolute index of the last letter of the match within the message")
    tag: Tag = Field(description="The tag associated to the message")


class Details(BaseModel, frozen=True):
    """A transaction and its corresponding tags at the entity-level."""

    content: str = Field(description="The message content itself")
    country_code: str | None = Field(default=None,
                                     description="The country code from which the message is."
                                                 " This can either be the GT, or a prediction.")
    country_code_confidence: float | None = Field(default=None,
                                                  description="The confidence of the country code,"
                                                              " if it represents a prediction")
    spans: list[TaggedSpan] = Field(description="A list of the spans of the message")

    def __str__(self):
        line_break = "\n"
        escaped_line_break = "\\n"
        return "\n".join(
            f"`{self.content[span.start:span.end].replace(line_break, escaped_line_break)}`: {span.tag.value}"
            for span in self.spans
        )

    def color_print(self, add_legend: bool = True):
        """
        Prints the current transaction with similar colors for each tag.
        Also prints the tags and their corresponding colors if `add_legend` is True.
        """
        # Local import to avoid a crash if the library is not installed
        import colored  # type: ignore

        tag_to_color = dict(zip(Tag, colored.library.Library.COLORS))

        color_used_str = set()

        for tagged_span in self.spans:
            print(f"{colored.back(tag_to_color[tagged_span.tag])}{self.content[tagged_span.start:tagged_span.end]}",
                  end="")
            color_used_str.add(f"{colored.back(tag_to_color[tagged_span.tag])}{tagged_span.tag.value}")

        if add_legend:
            print(colored.Style.reset
                  + "\nLegend: <"
                  + f"{colored.Style.reset}; ".join(color_used_str)
                  + colored.Style.reset
                  + ">"
                  + "\n")


class TokenizedDetails(Details, frozen=True):
    """A tokenized transaction and its corresponding bio-tags at the token level."""

    ids: list[int] = Field(description="The IDS of the tokens")
    spans: list[BIOTag] = Field(description="A list of the spans of the message")
