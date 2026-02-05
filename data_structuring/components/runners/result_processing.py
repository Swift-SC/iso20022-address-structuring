"""
Module providing the result classes for the runners.
"""
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, RootModel, Field

from data_structuring.components.details import Details, TaggedSpan
from data_structuring.components.fuzzy_matching import fuzzy_scan
from data_structuring.components.post_code_matching import post_code_match
from data_structuring.components.tags import Tag
from data_structuring.components.types import TorchTensor


class PredictionCRF(TaggedSpan):
    confidence: float = Field(description="The confidence of the prediction in [0, 1]")
    prediction: str = Field(description="The prediction itself")


class ResultRunnerCRF(BaseModel):
    details: Details = Field(description="An annotated `Details` object that contains detailed"
                                         " information about the prediction")
    predictions_per_tag: dict[Tag, set[PredictionCRF]] = Field(
        description="A dictionary mapping each tag to a list containing all entities tagged as that tag by the CRF"
    )
    emissions_per_tag: dict[Tag, TorchTensor] = Field(
        description="A dictionary mapping each tag to a tensor of shape (SEQUENCE_LENGTH) containing "
                    "the emission value of each character produced by the transformer for that tag"
    )
    log_probas_per_tag: dict[Tag, TorchTensor] = Field(
        description="A dictionary mapping each tag to a tensor of shape (SEQUENCE_LENGTH) containing "
                    "the log-probas value of each character for that tag"
    )


class ResultRunnerFuzzyMatch(BaseModel):
    country_matches: fuzzy_scan.FuzzyMatchResult = Field(description="A list of all `FuzzyMatch` "
                                                                     "detected for the countries")
    country_code_matches: fuzzy_scan.FuzzyMatchResult = Field(description="A list of all `FuzzyMatch` "
                                                                          "detected for the country codes")
    town_matches: fuzzy_scan.FuzzyMatchResult = Field(description="A list of all `FuzzyMatch` detected for the towns")
    extended_town_matches: fuzzy_scan.FuzzyMatchResult = Field(description="A list of all `FuzzyMatch` "
                                                                           "detected for the extended towns")


class ResultRunnerPostcodeMatch(BaseModel):
    postcode_matches: post_code_match.PostcodeMatchResult


def _to_str(obj, depth: int = 0) -> str:
    match obj:
        case list():
            if depth == 0:
                return "\n".join(_to_str(o, depth + 1) for o in obj)
            return "[" + ", ".join(_to_str(o, depth + 1) for o in obj) + "]"
        case dict():
            return "[" + ", ".join(
                f"{_to_str(key, depth + 1)}={_to_str(value, depth + 1)}" for key, value in obj.items()) + "]"
        case BaseModel() | RootModel():
            return _to_str(obj.model_dump(), depth + 1)
        case StrEnum():
            return obj.value
        case _:
            return str(obj)


class ResultPostProcessing(BaseModel):
    crf_result: ResultRunnerCRF = Field(description="The result of the CRF runner on the message")
    fuzzy_match_result: ResultRunnerFuzzyMatch = Field(description="The result of the fuzzymatch runner on the message")
    ibans: list[str] = Field(description="A list of the IBANs that have been identified in the message")
    suggested_country: str | None = Field(default=None, description="The suggested country code from the input")
    force_suggested_country: bool = Field(default=False,
                                          description="Whether the suggested country was forced as the only possibility")

    def _i_th_best_match(self,
                         i: int,
                         town_or_country: Literal[Tag.COUNTRY, Tag.TOWN],
                         value_if_none: Any | None = None):
        if town_or_country == Tag.COUNTRY:
            matches = self.fuzzy_match_result.country_matches
            resolved_attr = "origin"  # We take the origin for the countries
        elif town_or_country == Tag.TOWN:
            matches = self.fuzzy_match_result.town_matches
            resolved_attr = "possibility"  # We take the possibility for the towns
        else:
            raise ValueError(f"`town_or_country` should be {Tag.COUNTRY} or {Tag.TOWN}, got {town_or_country}.")

        if len(matches) > i:
            i_th_match = matches[i]
            return (getattr(i_th_match, resolved_attr), i_th_match.final_score,
                    getattr(i_th_match, "matched").replace('\n', ''))

        return value_if_none, value_if_none, value_if_none

    def i_th_best_match_country(self, i: int, value_if_none: Any = ""):
        return self._i_th_best_match(i=i, town_or_country=Tag.COUNTRY, value_if_none=value_if_none)

    def i_th_best_match_town(self, i: int, value_if_none: Any = ""):
        return self._i_th_best_match(i=i, town_or_country=Tag.TOWN, value_if_none=value_if_none)

    @staticmethod
    def save_list_as_json(list_results: list[ResultPostProcessing],
                          file_name: Path | str = "data_structuring_output.json",
                          **kwargs_model_dump_json
                          ) -> Path:
        """
        Save a list of `ResultPostProcessing` as a JSON.
        """

        # Define a temporary custom type to handle a list of `ResultPostProcessing` as
        # a pydantic model directly so that `model_dump_json` can be used
        _root_list_of_result_post_processing = RootModel[list[ResultPostProcessing]]
        json_str = _root_list_of_result_post_processing(list_results).model_dump_json(**kwargs_model_dump_json)

        path = Path(file_name)
        path.write_text(json_str, encoding="utf-8")

        return path

    @staticmethod
    def save_list_as_human_readable_csv(list_results: list["ResultPostProcessing"],
                                        file_name: Path | str = "data_structuring_output.csv",
                                        show_inferred_country: bool = False,
                                        n_best_matches: int = 2,
                                        confidence_precision: int = 2,
                                        verbose: bool = False,
                                        verbose_crf_tags: list[Tag] | None = None
                                        ) -> tuple[pl.DataFrame, Path]:
        """
        Save a list of `ResultPostProcessing` as a CSV, in a human friendly manner.
        """
        if verbose_crf_tags is None:
            verbose_crf_tags = [Tag.COUNTRY, Tag.TOWN, Tag.POSTAL_CODE]

        rows = []
        for result in list_results:

            # Start the row creation
            current_row = {"address": result.crf_result.details.content}

            if result.suggested_country:
                current_row["suggested_country"] = result.suggested_country
                current_row["force_suggested_country"] = str(result.force_suggested_country)

            # Add all best country/town matches, with their respective confidence
            for i in range(n_best_matches):
                (
                    best_match_country,
                    best_confidence_country,
                    best_match_raw_country
                ) = result.i_th_best_match_country(i=i)
                (
                    best_match_town,
                    best_confidence_town,
                    best_match_raw_town
                ) = result.i_th_best_match_town(i=i)

                current_row[f"{i + 1}th_best_country"] = best_match_raw_country
                if isinstance(best_confidence_country, float):
                    current_row[
                        f"{i + 1}th_best_country_confidence"] = (
                        f"{round(best_confidence_country * 100, confidence_precision)}%")
                    current_row[
                        f"{i + 1}th_best_country_resolved_code"] = best_match_country
                    if show_inferred_country:
                        current_row[f"{i + 1}th_inferred_country_resolved_code"] = (
                            result.fuzzy_match_result.town_matches[i].origin)
                else:  # No confidence (i.e, no detection)
                    current_row[f"{i + 1}th_best_country_confidence"] = best_confidence_country

                current_row[f"{i + 1}th_best_town"] = best_match_raw_town
                if isinstance(best_confidence_town, float):
                    current_row[
                        f"{i + 1}th_best_town_confidence"] = (
                        f"{round(best_confidence_town * 100, confidence_precision)}%")
                    current_row[
                        f"{i + 1}th_best_town_resolved"] = best_match_town
                else:  # No confidence (i.e, no detection)
                    current_row[f"{i + 1}th_best_town_confidence"] = best_confidence_town

            # Add verbose CRF information if needed
            if verbose:
                # Detailed fuzzy matches of towns and countries
                current_row["detailed_country_matches"] = _to_str(
                    result.fuzzy_match_result.country_matches.model_dump(mode="json"))
                current_row["detailed_town_matches"] = _to_str(
                    result.fuzzy_match_result.town_matches.model_dump(mode="json"))

                # Detailed CRF predictions for the requested tags
                for tag in verbose_crf_tags:
                    current_row[f"crf_prediction_{tag.value.lower()}"] = _to_str(
                        sorted(list(result.crf_result.predictions_per_tag[tag]), key=lambda x: x.start)
                    )

                # Detailed CountryHead predictions for the requested tags
                if result.crf_result.details.country_code is not None:
                    current_row["country_head_prediction"] = result.crf_result.details.country_code
                    current_row["country_head_confidence"] = (
                        f"{round(result.crf_result.details.country_code_confidence * 100, confidence_precision)}%")
                else:
                    current_row["country_head_prediction"] = "Country head disabled"
                    current_row["country_head_confidence"] = "Country head disabled"

                # Raw CRF spans predictions
                current_row["crf_spans"] = _to_str(result.crf_result.details)

                # IBANS
                current_row["ibans"] = _to_str(result.ibans)

            rows.append(current_row)

        final_df = pl.DataFrame(rows)
        file_name = Path(file_name)
        final_df.write_csv(file_name, separator="\t" if file_name.suffix == ".tsv" else ",")

        return final_df, file_name
