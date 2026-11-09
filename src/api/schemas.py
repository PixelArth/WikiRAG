from pydantic import BaseModel, Field

class Question(BaseModel):
    question: str = Field(min_length=8, max_length=512)

class Source(BaseModel):
    id: str | None
    text: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
