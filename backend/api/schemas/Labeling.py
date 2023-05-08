from pydantic import BaseModel


class LablingModel(BaseModel):
    theme: str
    elementId: str = None
    phrase: str = None
    positive: int = 1
    pattern: str = None

