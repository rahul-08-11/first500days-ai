import json
import logging
logger = logging.getLogger(__name__)


class SessionMemory:
    def __init__(self, max_turns: int = 6):

        self.sessions = {}
        self.max_turns = max_turns
        logging.info("SessionMemory initialized with max_turns=%d", max_turns)

    def get_session(self, session_id: str) -> list:
        """
        Returns message history in OpenAI-compatible format
        """
        return self.sessions.get(session_id, [])

    def add_user_message(self, session_id: str, content: str):
        self._add_message(session_id, "user", content)

    def add_assistant_message(self, session_id: str, content: str):
        self._add_message(session_id, "assistant", content)

    def _add_message(self, session_id: str, role: str, content: str):
        """
        Internal method to add a message to a session.

        Args:
            session_id (str): Unique identifier for the session.
            role (str): "user" or "assistant".
            content (str): The content of the message.

        Notes:
            Keeps only last N turns (user+assistant = 2 messages per turn).
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "role": role,
            "content": content
        })

        # Keep only last N turns (user+assistant = 2 messages per turn)
        self.sessions[session_id] = self.sessions[session_id][-self.max_turns * 2:]
