"""
Module providing the `TorchTensor` type for Pydantic use.
"""
from typing import Any

import torch
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class TorchTensor(torch.Tensor):
    """
    A torch.Tensor type implementation that is compatible with pydantic.
    """
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema(
                # We want the tensor to be converted to a list when being dumped to JSON
                serialization=core_schema.plain_serializer_function_ser_schema(
                    lambda t: t.tolist(),
                    when_used="json"
                )
            ),
        )

    @classmethod
    def validate(cls, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        raise ValueError(f"Expected torch.Tensor, got {type(value)}")
