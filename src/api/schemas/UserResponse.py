from pydantic import BaseModel


class UserResponse(BaseModel):
    advice: str
