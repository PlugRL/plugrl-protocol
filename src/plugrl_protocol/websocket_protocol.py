import enum

SERVER_STOP_REASON = "plugrl-server-stop"
SERVER_RESYNC_REASON = "plugrl-server-resync"


class MessageType(enum.Enum):
    INFER = "infer"
    FEEDBACK = "feedback"
    METADATA = "metadata"
    ACTION = "action"

    def __str__(self) -> str:
        return self.value
