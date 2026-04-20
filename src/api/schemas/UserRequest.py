from pydantic import BaseModel


class UserRequest(BaseModel):
    habit: str
