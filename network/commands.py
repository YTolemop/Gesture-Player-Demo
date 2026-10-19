from dataclasses import dataclass, asdict

LABELS = frozenset(("NONE", "MOVE_LEFT", "MOVE_RIGHT", "JUMP", "CROUCH",
                    "BLOCK", "NORMAL_ATTACK", "ATTACK_SKILL", "DEFENSE_SKILL",
                    "CONTROL_SKILL", "ULTIMATE"))


@dataclass(frozen=True)
class Commands:
    left: str = "NONE"
    right: str = "NONE"
    both: str = "NONE"

    def __post_init__(self):
        if any(v not in LABELS for v in (self.left, self.right, self.both)):
            raise ValueError("Unknown arena command")
        if self.both != "NONE" and (self.left != "NONE" or self.right != "NONE"):
            raise ValueError("both must be exclusive of left/right")

    def payload(self):
        return asdict(self)
