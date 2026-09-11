"""Caller-owned, bounded scratch for a single resident rank-merged wave.

One plan belongs to one buffer slot. Normal callers fence reuse by epoch_drained;
the audited service binding may use a fence dominating all scratch readers.
Externally visible completion events are never cached. There is no global
registry and no implicit synchronization on a cache miss.
"""
from contextlib import contextmanager
from functools import wraps
from threading import Lock


class FP8StreamingPreparedPlan:
    def __init__(self, *, device_wait: bool = False, static_input: bool = False, cache_wave_config: bool = False, workspace_bundle: bool = False, entry_descriptor: bool = False, scheduler_descriptor: bool = False):
        if not isinstance(device_wait, bool):
            raise TypeError("device_wait must be bool")
        if not isinstance(static_input, bool):
            raise TypeError("static_input must be bool")
        if static_input and not device_wait:
            raise ValueError("static_input requires device_wait")
        if not isinstance(cache_wave_config, bool):
            raise TypeError("cache_wave_config must be bool")
        if cache_wave_config and not static_input:
            raise ValueError("cache_wave_config requires static_input")
        if not isinstance(workspace_bundle, bool):
            raise TypeError("workspace_bundle must be bool")
        if workspace_bundle and not cache_wave_config:
            raise ValueError("workspace_bundle requires cache_wave_config")
        if not isinstance(entry_descriptor, bool):
            raise TypeError("entry_descriptor must be bool")
        if entry_descriptor and not workspace_bundle:
            raise ValueError("entry_descriptor requires workspace_bundle")
        if not isinstance(scheduler_descriptor, bool):
            raise TypeError("scheduler_descriptor must be bool")
        if scheduler_descriptor and not entry_descriptor:
            raise ValueError("scheduler_descriptor requires entry_descriptor")
        self._scheduler_descriptor = scheduler_descriptor
        self._entry_descriptor = entry_descriptor
        self._workspace_bundle = workspace_bundle
        self._cache_wave_config = cache_wave_config
        self._static_input = static_input
        self._device_wait = device_wait
        self._lock = Lock()
        self._owner = None
        self._signature = None
        self._resources = {}
        self._completion = None
        self._active = False
        self._poisoned = False
        self.generations = 0

    @property
    def scheduler_descriptor(self):
        return self._scheduler_descriptor

    @property
    def entry_descriptor(self):
        return self._entry_descriptor

    @property
    def workspace_bundle(self):
        return self._workspace_bundle

    @property
    def cache_wave_config(self):
        return self._cache_wave_config

    @property
    def static_input(self):
        return self._static_input

    @property
    def device_wait(self):
        """Queue the single full wave behind device-side pack completion waits."""
        return self._device_wait

    @property
    def resource_count(self):
        return len(self._resources)

    @contextmanager
    def lease(self, owner, signature):
        with self._lease(owner, signature, busy_ok=False) as plan:
            yield plan

    @contextmanager
    def try_lease(self, owner, signature):
        """Atomically select and lease; busy is None, invalid/poisoned still fails.

        The completion is queried once while holding the submission lock. A
        caller must keep this context open through launch and finish.
        """
        with self._lease(owner, signature, busy_ok=True) as plan:
            yield plan

    @contextmanager
    def _lease(self, owner, signature, *, busy_ok):
        if not self._lock.acquire(blocking=False):
            if busy_ok:
                yield None
                return
            raise RuntimeError("prepared plan is already in use by another submission")
        entered = False
        try:
            if self._poisoned:
                raise RuntimeError("prepared plan had a failed submission; do not reuse it")
            if self._completion is not None and not self._completion.query():
                if busy_ok:
                    yield None
                    return
                raise RuntimeError("prepared plan's previous epoch has not drained")
            if self._owner is not None and self._owner is not owner:
                raise ValueError("prepared plan belongs to a different buffer slot")
            if self._signature is not None and self._signature != signature:
                raise ValueError("prepared plan shape/configuration changed; use a new plan")
            self._owner = owner
            self._signature = signature
            self._active = entered = True
            yield self
        except BaseException:
            if entered:
                # A partial submission may still be writing scratch. Preserve
                # resources and never interpret the old completion as its fence.
                self._poisoned = True
            raise
        finally:
            self._active = False
            self._lock.release()

    def resource(self, name, spec, create):
        if not self._active:
            raise RuntimeError("prepared resources require an active plan lease")
        if name not in self._resources:
            self._resources[name] = (spec, create())
        cached_spec, value = self._resources[name]
        if cached_spec != spec:
            raise ValueError(f"prepared resource {name} changed shape/configuration")
        return value

    def finish(self, completion):
        if not self._active:
            raise RuntimeError("prepared completion requires an active plan lease")
        self._completion = completion
        self.generations += 1


def prepared_submission(fn):
    """Guard reuse across the whole submission, including early allocations."""
    @wraps(fn)
    def launch(dispatch, *args, **kwargs):
        plan = kwargs.get("prepared_plan")
        if plan is None:
            return fn(dispatch, *args, **kwargs)
        if not isinstance(plan, FP8StreamingPreparedPlan):
            raise TypeError("prepared_plan must be FP8StreamingPreparedPlan")
        # Scratch depends on activation metadata. Weight shape validation stays
        # in the runner; weights and generation-varying input pointers aren't cached.
        tensors = (dispatch.x, dispatch.sf, dispatch.expert_psum)
        signature = tuple(
            (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
            if t is not None else None for t in tensors
        )
        with plan.lease(dispatch.buffer, signature):
            result = fn(dispatch, *args, **kwargs)
            plan.finish(result.epoch_drained)
            return result
    return launch
