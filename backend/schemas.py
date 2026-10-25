from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    screening_result: Optional[str]
    dyslexic: Optional[bool]
    screening_score: Optional[float] = Field(default=None, ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    disclaimer: str
    line_count: int = Field(ge=0)
    crop_count: int = Field(ge=0)
    class_counts: dict[str, int]
    artifacts: dict[str, str]
    processing_time_ms: float = Field(ge=0)
    segmentation_metadata: Optional[dict] = None


class GuidanceRequest(BaseModel):
    status: str
    screening_score: Optional[float] = Field(default=None, ge=0, le=1)
    activeThreshold: float = Field(ge=0, le=1)
    crop_count: int = Field(ge=0)
    line_count: int = Field(ge=0)
    class_counts: dict[str, int]
    age_group: Literal["Child", "Teen", "Adult"] = "Child"
    difficulty: Literal["Gentle", "Standard", "Challenge"] = "Standard"
