from dataclasses import dataclass


@dataclass
class CommitResult:
    committed: bool
    text: str
    stable_label: str | None


class DebounceAccumulator:
    def __init__(self, min_stable_frames: int = 8, cooldown_frames: int = 10):
        self.min_stable_frames = min_stable_frames
        self.cooldown_frames = cooldown_frames
        self.current_label: str | None = None
        self.stable_frames = 0
        self.cooldown = 0
        self.text = ""

    def force_commit(self, label: str) -> CommitResult:
        """J/Z처럼 동적 제스처는 감지 즉시 확정."""
        if self.cooldown > 0:
            return CommitResult(False, self.text, label)
        self.text += label
        self.cooldown = self.cooldown_frames
        self.current_label = None
        self.stable_frames = 0
        return CommitResult(True, self.text, label)

    def update(self, label: str | None) -> CommitResult:
        if self.cooldown > 0:
            self.cooldown -= 1

        if label is None:
            self.current_label = None
            self.stable_frames = 0
            return CommitResult(False, self.text, None)

        if label == self.current_label:
            self.stable_frames += 1
        else:
            self.current_label = label
            self.stable_frames = 1

        if self.stable_frames >= self.min_stable_frames and self.cooldown == 0:
            self.text += label
            self.cooldown = self.cooldown_frames
            self.stable_frames = 0
            return CommitResult(True, self.text, label)

        return CommitResult(False, self.text, self.current_label)

    def clear(self) -> None:
        self.current_label = None
        self.stable_frames = 0
        self.cooldown = 0
        self.text = ""

    def append_space(self) -> None:
        self.text += " "
        self.current_label = None
        self.stable_frames = 0

    def backspace(self) -> None:
        self.text = self.text[:-1]
        self.current_label = None
        self.stable_frames = 0
