from typing import Any

from pydantic import BaseModel


class ResponseModal(BaseModel):
    message: str
    success: bool
    data: Any | None

