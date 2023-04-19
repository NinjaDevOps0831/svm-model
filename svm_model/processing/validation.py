from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from svm_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.dropna()
    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data[config.model_config.selected_vars].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleUserDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitannicDataInputSchema(BaseModel):
    Pclass: Optional[int]
    Sex: Optional[str]
    Age: Optional[int]
    SibSp: Optional[int]
    Parch: Optional[int]
    Fare: Optional[float]
    Embarked: Optional[str]


class MultipleUserDataInputs(BaseModel):
    inputs: List[TitannicDataInputSchema]
