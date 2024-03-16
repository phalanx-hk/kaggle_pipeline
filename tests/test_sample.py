import pytest
from pydantic import Field, BaseModel, model_validator


class Bbox(BaseModel):
    x1: float = Field(..., ge=0, le=1)
    x2: float = Field(..., ge=0, le=1)
    y1: float = Field(..., ge=0, le=1)
    y2: float = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def validate_bbox_coordinates(self):
        if self.x1 >= self.x2:
            raise ValueError("left coordinate must be smaller than right coordinate")
        if self.y1 >= self.y2:
            raise ValueError("top coordinate must be smaller than bottom coordinate")
        return self


@pytest.mark.parametrize(
    "bbox",
    [
        {"x1": 0.1, "x2": 0.2, "y1": 0.1, "y2": 0.2},
        {"x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0},
    ],
)
def test_validate_bbox(bbox: dict[str, float]):
    Bbox(**bbox)


@pytest.mark.parametrize(
    "bbox",
    [
        {"x1": -0.1, "x2": 0.2, "y1": 0.1, "y2": 0.2},
        {"x1": 0.2, "x2": -0.1, "y1": 0.1, "y2": 0.2},
        {"x1": 0.1, "x2": 0.2, "y1": 1.2, "y2": 0.1},
        {"x1": 0.2, "x2": 0.1, "y1": 0.2, "y2": 100000},
    ],
)
def test_invalid_bbox(bbox: dict[str, float]):
    with pytest.raises(ValueError):
        Bbox(**bbox)
