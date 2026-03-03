"""可观测性流式运行时：事件分发与订阅。"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional
import pickle

from ...core.observability import (
    EVENT_TYPE_SUBSCRIBER_ERRORED,
    ObsEventEnvelope,
    ObsStartPosition,
    ObsSubscriptionOptions,
    ObsSubscriptionState,
    ObsSubscriptionStatus,
)
from .history_store import SQLiteHistoryStore
from .subscriber_state import SubscriberState, estimate_event_size_bytes

_DEFAULT_FETCH_CHUNK = 128


def _is_pickleable(value: Any) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:
        return False


def _normalize_payload_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalize_payload_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_payload_value(v) for v in value]
    if _is_pickleable(value):
        return value
    return repr(value)


class ObsStreamRuntime:
    """流式订阅运行时。"""

    def __init__(self, history_dir: str, keep_history_files: bool, default_max_memory_bytes: int) -> None:
        self._history = SQLiteHistoryStore(history_dir)
        self._keep_history_files = bool(keep_history_files)
        self._default_max_memory_bytes = int(max(1, default_max_memory_bytes))
        self._seq = 0
        self._run_id: Optional[str] = None
        self._run_finished = False
        self._subscribers: Dict[str, SubscriberState] = {}
        self._lock = threading.Lock()
        self._pending: Deque[ObsEventEnvelope] = deque()
        self._pending_cond = threading.Condition()
        self._dispatcher_stop = False
        self._dispatcher_thread: Optional[threading.Thread] = None

    def start_run(self, run_id: Optional[str] = None) -> str:
        with self._lock:
            if self._run_finished and not self._keep_history_files:
                self._history.cleanup()
            assigned_run_id = run_id or uuid.uuid4().hex
            self._close_all_subscribers_locked()
            self._run_id = assigned_run_id
            self._seq = 0
            self._run_finished = False
            self._dispatcher_stop = False
            self._pending.clear()
            self._history.start_run(assigned_run_id)
            thread = threading.Thread(target=self._dispatcher_loop, name=f"obs-dispatch-{assigned_run_id}")
            thread.daemon = True
            self._dispatcher_thread = thread
            thread.start()
            return assigned_run_id

    def finish_run(self) -> None:
        with self._pending_cond:
            self._dispatcher_stop = True
            self._pending_cond.notify_all()
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5.0)
        with self._lock:
            self._run_finished = True
            self._history.close_writer()

    def publish(self, event_type: str, sim_time: int, payload: Mapping[str, object]) -> ObsEventEnvelope:
        with self._lock:
            if self._run_id is None:
                raise RuntimeError("Run is not started")
            self._seq += 1
            event = ObsEventEnvelope(
                run_id=self._run_id,
                seq=self._seq,
                event_type=str(event_type),
                sim_time=int(sim_time),
                wall_time=float(time.time()),
                payload=_normalize_payload_value(dict(payload)),
            )
        with self._pending_cond:
            self._pending.append(event)
            self._pending_cond.notify()
        return event

    def subscribe(self, options: Optional[ObsSubscriptionOptions] = None) -> str:
        cfg = options or ObsSubscriptionOptions()
        with self._lock:
            if self._run_finished and not self._history.has_history():
                raise RuntimeError("History has been cleaned up")
            gate_seq = self._seq
            replay_seq = 1 if cfg.start_position == ObsStartPosition.BEGINNING else gate_seq + 1
            topics = set(cfg.topics) if cfg.topics else None
            max_memory = int(cfg.max_memory_bytes) if cfg.max_memory_bytes > 0 else self._default_max_memory_bytes
            sub_id = uuid.uuid4().hex
            self._subscribers[sub_id] = SubscriberState(
                subscription_id=sub_id,
                topics=topics,
                max_memory_bytes=max_memory,
                live_min_seq=gate_seq + 1,
                replay_seq=replay_seq,
            )
            return sub_id

    def poll(self, subscription_id: str, max_items: int, timeout_ms: int) -> List[ObsEventEnvelope]:
        limit = max(1, int(max_items))
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)
        deadline = time.time() + timeout_s
        out: List[ObsEventEnvelope] = []
        while len(out) < limit:
            sub = self._get_subscriber(subscription_id)
            self._ensure_subscriber_readable(sub)
            out.extend(self._drain_replay(sub, limit - len(out)))
            if len(out) >= limit:
                return out
            out.extend(self._drain_live_buffer(sub, limit - len(out)))
            if len(out) >= limit or timeout_s <= 0.0:
                return out
            wait_left = deadline - time.time()
            if wait_left <= 0:
                return out
            with sub.condition:
                sub.condition.wait(wait_left)
        return out

    def unsubscribe(self, subscription_id: str) -> None:
        sub = self._get_subscriber(subscription_id)
        with sub.condition:
            sub.state = ObsSubscriptionState.CLOSED
            sub.buffer.clear()
            sub.buffer_bytes = 0
            sub.condition.notify_all()
        with self._lock:
            self._try_cleanup_history_locked()

    def get_subscription_status(self, subscription_id: str) -> ObsSubscriptionStatus:
        sub = self._get_subscriber(subscription_id)
        with sub.condition:
            return ObsSubscriptionStatus(
                subscription_id=sub.subscription_id,
                state=sub.state,
                error=sub.error,
                buffered_events=sub.buffer_len(),
                buffered_bytes=sub.buffer_bytes,
            )

    def _get_subscriber(self, subscription_id: str) -> SubscriberState:
        with self._lock:
            sub = self._subscribers.get(subscription_id)
        if sub is None:
            raise KeyError(f"Unknown subscription_id: {subscription_id}")
        return sub

    @staticmethod
    def _ensure_subscriber_readable(sub: SubscriberState) -> None:
        if sub.state == ObsSubscriptionState.ERRORED:
            raise RuntimeError(sub.error or "Subscriber errored")
        if sub.state == ObsSubscriptionState.CLOSED:
            raise RuntimeError("Subscriber closed")

    def _drain_replay(self, sub: SubscriberState, max_items: int) -> List[ObsEventEnvelope]:
        if max_items <= 0:
            return []
        with sub.condition:
            start_seq = sub.replay_seq
            end_seq = sub.live_min_seq - 1
        if start_seq > end_seq:
            return []
        batch_limit = max(max_items * 4, _DEFAULT_FETCH_CHUNK)
        events = self._history.fetch_range(start_seq, end_seq, batch_limit)
        if not events:
            with sub.condition:
                if sub.replay_seq <= end_seq:
                    sub.replay_seq = end_seq + 1
            return []
        out: List[ObsEventEnvelope] = []
        with sub.condition:
            for event in events:
                sub.replay_seq = int(event.seq) + 1
                if sub.matches(event.event_type):
                    out.append(event)
                    if len(out) >= max_items:
                        break
        return out

    @staticmethod
    def _drain_live_buffer(sub: SubscriberState, max_items: int) -> List[ObsEventEnvelope]:
        if max_items <= 0:
            return []
        out: List[ObsEventEnvelope] = []
        with sub.condition:
            while sub.buffer and len(out) < max_items:
                event, size = sub.buffer.popleft()
                sub.buffer_bytes = max(0, sub.buffer_bytes - int(size))
                out.append(event)
        return out

    def _dispatcher_loop(self) -> None:
        while True:
            with self._pending_cond:
                while not self._pending and not self._dispatcher_stop:
                    self._pending_cond.wait()
                if not self._pending and self._dispatcher_stop:
                    return
                event = self._pending.popleft()
            self._history.append(event)
            self._dispatch_to_subscribers(event)

    def _dispatch_to_subscribers(self, event: ObsEventEnvelope) -> None:
        with self._lock:
            subscribers = list(self._subscribers.values())
        for sub in subscribers:
            self._enqueue_to_subscriber(sub, event)

    def _enqueue_to_subscriber(self, sub: SubscriberState, event: ObsEventEnvelope) -> None:
        with sub.condition:
            if sub.state != ObsSubscriptionState.ACTIVE or int(event.seq) < int(sub.live_min_seq):
                return
            if not sub.matches(event.event_type):
                return
            size = estimate_event_size_bytes(event)
            if sub.buffer_bytes + size > sub.max_memory_bytes:
                sub.state = ObsSubscriptionState.ERRORED
                sub.error = "Subscriber buffer exceeded max_memory_bytes"
                sub.buffer.clear()
                sub.buffer_bytes = 0
                sub.condition.notify_all()
                self._publish_subscriber_error(sub.subscription_id, sub.error)
                with self._lock:
                    self._try_cleanup_history_locked()
                return
            sub.buffer.append((event, size))
            sub.buffer_bytes += size
            sub.condition.notify_all()

    def _publish_subscriber_error(self, subscription_id: str, reason: str) -> None:
        try:
            self.publish(
                event_type=EVENT_TYPE_SUBSCRIBER_ERRORED,
                sim_time=self._seq,
                payload={"subscription_id": subscription_id, "reason": reason},
            )
        except Exception:
            return

    def _close_all_subscribers_locked(self) -> None:
        for sub in self._subscribers.values():
            with sub.condition:
                sub.state = ObsSubscriptionState.CLOSED
                sub.buffer.clear()
                sub.buffer_bytes = 0
                sub.condition.notify_all()
        self._subscribers.clear()

    def _try_cleanup_history_locked(self) -> None:
        if self._keep_history_files or not self._run_finished:
            return
        has_active = any(sub.state == ObsSubscriptionState.ACTIVE for sub in self._subscribers.values())
        if has_active:
            return
        self._history.cleanup()
