from typing import TypedDict, Optional

class InputState(TypedDict):
    protocolVersion: int
    sessionId: str
    matchId: Optional[str]
    seq: int
    gesture: str
    generation: int
