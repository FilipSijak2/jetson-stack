from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import websocket


class RosbridgeClient:
    def __init__(self, url: str, logger: logging.Logger) -> None:
        self.url = url
        self.logger = logger
        self.ws: Optional[websocket.WebSocket] = None

    def connect(self) -> None:
        self.logger.info("Connecting to rosbridge at %s", self.url)
        self.ws = websocket.create_connection(self.url, timeout=5)
        self.ws.settimeout(0.2)
        self.logger.info("Connected to rosbridge")

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None

    def send(self, payload: Dict[str, Any]) -> None:
        if self.ws is None:
            raise RuntimeError("rosbridge websocket is not connected")
        self.ws.send(json.dumps(payload, separators=(",", ":")))

    def advertise(self, topic: str, msg_type: str) -> None:
        self.send({"op": "advertise", "topic": topic, "type": msg_type})

    def subscribe(
        self,
        topic: str,
        msg_type: str,
        queue_length: int = 1,
        throttle_rate: int = 0,
    ) -> None:
        self.send(
            {
                "op": "subscribe",
                "topic": topic,
                "type": msg_type,
                "queue_length": queue_length,
                "throttle_rate": throttle_rate,
            }
        )

    def publish(self, topic: str, msg: Dict[str, Any]) -> None:
        self.send({"op": "publish", "topic": topic, "msg": msg})

    def recv_json(self) -> Optional[Dict[str, Any]]:
        if self.ws is None:
            raise RuntimeError("rosbridge websocket is not connected")
        try:
            raw = self.ws.recv()
        except websocket.WebSocketTimeoutException:
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.warning("Ignoring invalid JSON from rosbridge: %s", exc)
            return None
        if not isinstance(payload, dict):
            self.logger.warning("Ignoring non-object rosbridge payload")
            return None
        return payload

