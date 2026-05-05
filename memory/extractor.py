from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from config import MEMORY_MODEL
from prompts.memory import MEMORY_PROMPT


class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")


class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)


memory_llm = ChatGroq(model=MEMORY_MODEL, temperature=0)
memory_extractor = memory_llm.with_structured_output(MemoryDecision)


def extract_memories(last_text: str, existing: str) -> MemoryDecision:
    return memory_extractor.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_text},
        ]
    )