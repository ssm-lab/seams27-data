"""Synchronous JSON request/reply client for the EMF adaptation engine."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from config_loader import EnvSpec


class EmfBridgeError(RuntimeError):
    """The engine rejected a request or returned an invalid response."""


class EmfBridge:
    """One socket for one training session; every request waits for its reply."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5558", timeout_ms: int = 30_000, *, generation_seed=123456, trace_path=None):
        import zmq

        self.generation_seed = generation_seed
        self._trace = None
        if trace_path is not None:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._trace = Path(trace_path).open("w")
        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(endpoint)

    def __enter__(self) -> "EmfBridge":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._socket.close()
        if self._trace is not None:
            self._trace.close()

    def initialize(self) -> EnvSpec:
        """Ask the engine to start a fresh curriculum and return its initial environment."""
        reply = self._request({"type": "initialize", "generationSeed": self.generation_seed}, "result")

        if reply.get("instruction") != "PROCEED_TO_ENVIRONMENT":
            raise EmfBridgeError(
                f"initialize expected PROCEED_TO_ENVIRONMENT, "
                f"received {reply.get('instruction')!r}"
            )

        return self.environment_spec_from_message(reply.get("environment"))

    def report_episode_outcome(self, *, episode: int, episodic_return: float, success: bool) -> dict[str, Any]:
        """Report one completed episode and return the engine's instruction."""
        if type(episode) is not int or episode <= 0:
            raise ValueError("episode must be a positive integer")

        if not math.isfinite(episodic_return):
            raise ValueError("episodic_return must be finite")

        adaptation_result = self._request(
            {"type": "episodeOutcome", "episode": episode,
             "episodicReturn": float(episodic_return), "success": bool(success)},
            "result",
        )

        if (type(adaptation_result.get("episode")) is not int or adaptation_result["episode"] != episode):
            raise EmfBridgeError(f"expected adaptation result for episode {episode}, "f"received {adaptation_result.get('episode')}")

        instruction = adaptation_result.get("instruction")

        if instruction not in ("CONTINUE", "PROCEED_TO_ENVIRONMENT", "STOP"):
            raise EmfBridgeError(f"unknown instruction: {instruction!r}")

        if instruction == "PROCEED_TO_ENVIRONMENT":
            adaptation_result["environment"] = (self.environment_spec_from_message(adaptation_result.get("environment")))

        elif "environment" in adaptation_result:
            raise EmfBridgeError("only PROCEED_TO_ENVIRONMENT may include an environment")

        return adaptation_result

    @staticmethod
    def environment_spec_from_message(environment: Any) -> EnvSpec:
        """Validate the existing BurningForest payload and convert it to EnvSpec."""
        if not isinstance(environment, dict):
            raise EmfBridgeError("environment must be an object")
        env_id = environment.get("env_id")
        rows = environment.get("desc")
        rewards = environment.get("reward_schedule")
        if not isinstance(env_id, str) or not env_id:
            raise EmfBridgeError("environment needs a non-empty env_id")
        if (not isinstance(rows, list) or not rows
                or any(not isinstance(row, str) or not row for row in rows)
                or any(len(row) != len(rows[0]) for row in rows)):
            raise EmfBridgeError("desc must be a non-empty rectangular list of strings")
        if (not isinstance(rewards, list) or len(rewards) != 5
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in rewards)):
            raise EmfBridgeError("reward_schedule must contain five finite numbers")
        return EnvSpec(env_id=env_id, desc=list(rows), reward_schedule=tuple(float(v) for v in rewards))

    def _request(self, message: dict[str, Any], expected_type: str) -> dict[str, Any]:
        try:
            self._socket.send_json(message)
            reply = self._socket.recv_json()
        except self._zmq.Again as error:
            # A REQ socket cannot send again while its previous reply is pending.
            self.close()
            raise TimeoutError(f"EMF runtime timed out during {message['type']}; session closed") from error
        except ValueError as error:
            raise EmfBridgeError("EMF runtime reply is not valid JSON") from error
        if self._trace is not None:
            self._trace.write(json.dumps({"request": message, "response": reply}) + "\n")
            self._trace.flush()
        if not isinstance(reply, dict):
            raise EmfBridgeError("EMF runtime reply must be an object")
        if reply.get("type") == "error":
            raise EmfBridgeError(str(reply.get("message", "engine rejected the request")))
        if reply.get("type") != expected_type:
            raise EmfBridgeError(f"expected {expected_type}, received {reply.get('type')}")
        return reply
