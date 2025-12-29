"""
Module providing utility functions for the models.
"""
import math
import torch

from data_structuring.components.tags import BIO, BIOTag, Tag
from data_structuring.components.details import Details, TaggedSpan


# Taken from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def pos_embed_1d(d_model: int, length: int):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def create_details_from_biotags(raw_content: str,
                                country: str,
                                country_confidence: float,
                                tags: list[BIOTag],
                                pad_with_other: bool = True,
                                strict_before_inside: bool = True,
                                ) -> Details | None:
    """
    Given a string, and the tags of each of its tokens, create a Details object.
    The BIOTags are first grouped into consistent spans (i.e., only valid sequences of <B-*, I-*, ...> are parsed),
    and then transformed into tags to create the Details object.

    Args:
        raw_content (str): The raw content of the transaction.
        tags (list[BIOTag]): The tags of each of its tokens.
        country_confidence (float): The confidence of the country code, if it represents a prediction.
        pad_with_other (bool, optional): Whether to pad the BIOTags with Tag.OTHER. Defaults to True.
        strict_before_inside (bool, optional): If true, a sequence of BIOTags must begin with a BEFORE tag. Otherwise,
         an INSIDE tag will be accepted.

    Returns:
        Details: The Details object that contains the transaction and its corresponding tags.
    """

    if len(tags) == 0:
        return None

    current_tag = tags[0]
    current_pos = 0

    # (1) Group BIO tags
    # Create tuples of ((start, end), tag) by grouping similar tags
    grouped_bio_tags = []
    for i in range(1, len(tags)):
        if tags[i] == current_tag:
            continue

        grouped_bio_tags.append(((current_pos, i), current_tag))
        current_tag = tags[i]
        current_pos = i

    grouped_bio_tags.append(((current_pos, len(tags)), current_tag))

    # (2) Transform BIO tags to tags
    # Strict mode: A span should start with a B-* tag followed by I-* tag(s)
    # Non-strict mode: A span is formed with all contiguous B-* or I-* tags of the same type
    grouped_spans = (_get_strict_span(grouped_bio_tags, pad_with_other)
                     if strict_before_inside
                     else _get_span(grouped_bio_tags))

    # Pad missing spans with Tag.OTHER
    if pad_with_other:
        previous_end = grouped_spans[-1].end if len(grouped_spans) > 0 else 0
        if previous_end != len(raw_content):
            grouped_spans.append(TaggedSpan(start=previous_end, end=len(raw_content), tag=Tag.OTHER))

    details = Details(content=raw_content, spans=grouped_spans, country_code=country,
                      country_code_confidence=country_confidence)
    return details


def _get_strict_span(grouped_bio_tags: list, pad_with_other: bool) -> list[TaggedSpan]:
    grouped_spans: list[TaggedSpan] = []

    for ((start, _), bio_tag_1), ((_, end), bio_tag_2) in zip(grouped_bio_tags, grouped_bio_tags[1:]):
        if bio_tag_1.tag == bio_tag_2.tag and bio_tag_1.bio == BIO.BEFORE and bio_tag_2.bio == BIO.INSIDE:
            # Pad missing spans between the last one and the current one with Tag.OTHER
            if pad_with_other:
                previous_end = grouped_spans[-1].end if len(grouped_spans) > 0 else 0
                if start != previous_end:
                    grouped_spans.append(TaggedSpan(start=previous_end, end=start, tag=Tag.OTHER))
            # Add the corresponding span
            grouped_spans.append(TaggedSpan(start=start, end=end, tag=bio_tag_1.tag))
    return grouped_spans


def _get_span(grouped_bio_tags: list) -> list[TaggedSpan]:

    if len(grouped_bio_tags) == 0:
        return []

    grouped_spans: list[TaggedSpan] = []
    (current_start, current_end), current_bio_tag = grouped_bio_tags[0]

    for (start, end), bio_tag in grouped_bio_tags:
        if bio_tag.tag == current_bio_tag.tag:
            # Same tag: We continue expanding the window
            current_end = end
        else:
            # Different tag: We cut the current window and add it to the list
            grouped_spans.append(TaggedSpan(start=current_start, end=current_end, tag=current_bio_tag.tag))
            # Creation of a new window
            current_start, current_end, current_bio_tag = start, end, bio_tag

    # Add the final remaining window
    grouped_spans.append(TaggedSpan(start=current_start, end=grouped_bio_tags[-1][0][1], tag=current_bio_tag.tag))
    return grouped_spans
