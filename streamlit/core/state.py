from collections import deque
from enum import IntEnum
from typing import Optional
import time


class AlertLevel(IntEnum):
    NORMAL = 0
    WATCH = 1
    WARNING = 2
    ALERT = 3

    @property
    def label(self) -> str:
        return self.name


class RollingBuffer:

    def __init__(self, window_size: int = 400):

        self.window_size = window_size

        self.buffer = deque(
            maxlen=window_size
        )


    def push(self, row: dict):

        self.buffer.append(row)


    def is_full(self) -> bool:

        return len(self.buffer) == self.window_size


    def as_list(self) -> list:

        return list(self.buffer)


    def __len__(self):

        return len(self.buffer)


class AlertStateMachine:

    def __init__(
        self,
        watch_threshold: float = 0.30,
        warning_threshold: float = 0.60,
        alert_threshold: float = 0.85,
        promote_after: int = 3,
        demote_after: int = 5,
    ):

        self.watch_threshold = watch_threshold
        self.warning_threshold = warning_threshold
        self.alert_threshold = alert_threshold

        self.promote_after = promote_after
        self.demote_after = demote_after

        self.current_level = AlertLevel.NORMAL

        self._consecutive_above = 0
        self._consecutive_below = 0

        self.last_updated = time.time()


    def _level_for_score(
        self,
        score: float,
    ) -> AlertLevel:

        if score >= self.alert_threshold:
            return AlertLevel.ALERT

        if score >= self.warning_threshold:
            return AlertLevel.WARNING

        if score >= self.watch_threshold:
            return AlertLevel.WATCH

        return AlertLevel.NORMAL


    def update(
        self,
        score: float,
    ) -> AlertLevel:

        target = self._level_for_score(score)

        if target > self.current_level:

            self._consecutive_above += 1
            self._consecutive_below = 0

            if self._consecutive_above >= self.promote_after:

                self.current_level = target
                self._consecutive_above = 0

        elif target < self.current_level:

            self._consecutive_below += 1
            self._consecutive_above = 0

            if self._consecutive_below >= self.demote_after:

                self.current_level = AlertLevel(
                    max(
                        0,
                        self.current_level - 1
                    )
                )

                self._consecutive_below = 0

        else:

            self._consecutive_above = 0
            self._consecutive_below = 0

        self.last_updated = time.time()

        return self.current_level


class SessionState:

    def __init__(
        self,
        window_size: int = 400,
    ):

        self.buffer = RollingBuffer(
            window_size=window_size
        )

        self.alert_machine = AlertStateMachine()

        self.is_replaying = False

        self.current_row_index = 0
        self.total_rows = 0

        self.latest_prediction: Optional[dict] = None

        self.selected_model = "random_forest"

        self.history = []


    def reset(self):

        self.__init__(
            window_size=self.buffer.window_size
        )