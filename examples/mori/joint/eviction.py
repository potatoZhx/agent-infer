"""Oracle deadline eviction over the real v0.31 free queue, without a shadow pool."""

from __future__ import annotations

import contextlib
import heapq
import math
import time
from collections import Counter, defaultdict, deque


class _MutationSet(set):
    """Preserve job metadata; notify only when source protection changes."""

    def __init__(self, values, changed):
        super().__init__(values)
        self.changed = changed

    def add(self, value):
        empty = not self
        super().add(value)
        if empty:
            self.changed(self)

    def remove(self, value):
        super().remove(value)
        if not self:
            self.changed(self)

    def discard(self, value):
        present = bool(self)
        super().discard(value)
        if present and not self:
            self.changed(self)

    def clear(self):
        if self:
            super().clear()
            self.changed(self)

    def pop(self):
        value = super().pop()
        if not self:
            self.changed(self)
        return value

    def _update(self, method, *values):
        before = bool(self)
        try:
            return getattr(super(), method)(*values)
        finally:
            # Native updates can partially consume an iterator before it raises.
            # A real protection transition still needs notification in that case.
            if bool(self) != before:
                self.changed(self)

    def update(self, *values):
        self._update("update", *values)

    def difference_update(self, *values):
        self._update("difference_update", *values)

    def intersection_update(self, *values):
        self._update("intersection_update", *values)

    def symmetric_difference_update(self, value):
        self._update("symmetric_difference_update", value)

    def __ior__(self, other):
        self.update(other)
        return self

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def __ixor__(self, other):
        self.symmetric_difference_update(other)
        return self


class _MutationDict(dict):
    """Track protection existence, preserving every native metadata mutation."""

    def __init__(self, values, changed, *, sets=False):
        self.changed, self.sets = changed, sets
        super().__init__((key, self._value(key, value)) for key, value in values.items())

    def _value(self, key, value):
        if not self.sets:
            return value

        def changed(current):
            # A set detached by replacement/pop can still have a native caller.
            # Its later mutations do not change the current map's protection.
            # Pass the set as an argument rather than capture it: capturing it
            # would create a per-set cycle that survives native pop/delete.
            if self.get(key) is current:
                self.changed(key)

        return _MutationSet(value, changed)

    def __setitem__(self, key, value):
        present = key in self
        before = bool(self[key]) if self.sets and present else present
        if self.sets and present and self[key] is value:
            return
        wrapped = self._value(key, value)
        super().__setitem__(key, wrapped)
        after = bool(wrapped) if self.sets else True
        if before != after:
            self.changed(key)

    def __delitem__(self, key):
        protected = bool(self[key]) if self.sets else key in self
        super().__delitem__(key)
        if protected:
            self.changed(key)

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        if key not in self:
            return super().pop(key, *default)
        value = super().pop(key)
        if not self.sets or value:
            self.changed(key)
        return value

    def popitem(self):
        value = super().popitem()
        if not self.sets or value[1]:
            self.changed(value[0])
        return value

    def clear(self):
        if self:
            protected = any(self.values()) if self.sets else True
            super().clear()
            if protected:
                self.changed(None)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __ior__(self, other):
        self.update(other)
        return self


class InsufficientEvictableBlocks(RuntimeError):
    pass


class GapFarthestEviction:
    """Select free prefix tails; native allocation still owns eviction/ref updates.

    Only observed request hashes enter this adapter. Physical duplicates count
    separately, whereas a shared content deadline is its earliest live demand.
    Native pending-store sources stay unavailable until their completion fence.
    """

    def __init__(self, scheduler, state, *, emit=None):
        self.scheduler, self.state = scheduler, state
        self.pool = scheduler.kv_cache_manager.block_pool
        self.queue = self.pool.free_block_queue
        self.usable_blocks = sum(not b.is_null for b in self.pool.blocks)
        connector = getattr(scheduler, "connector", None)
        self.offload = getattr(connector, "scheduler", None)
        self.emit = emit or (lambda row: None)
        self.on_cache_change = lambda: None
        self.parents, self.depth = {}, {}
        self.requests, self.chains = {}, {}
        self.content_owners = defaultdict(set)
        self.finished = set()
        self._forced = None
        self._forced_info = None
        self._last_preview = ((), "none", 0)
        self.stats = Counter()
        self.seconds = 0.0
        self.thread_cpu_seconds = 0.0
        self.cache_epoch = 0
        self.location_epoch = 0
        self._available_cache = None
        self._structure_cache = None
        self._mutation_hooks = []
        self._tracked_mappings = {}
        self._previous_epoch_hook = getattr(self.pool, "_joint_cache_epoch", None)
        self._previous_location_epoch_hook = getattr(self.pool, "_joint_location_epoch", None)
        self._original_pop = self.queue.popleft_n
        self._original_free_count = self.pool.get_num_free_blocks
        self._original_lookup = scheduler.kv_cache_manager.get_computed_blocks
        self._original_allocate = self.pool.get_new_blocks
        # The native pool remains authoritative. These indexes summarize only
        # its current physical blocks and are updated from native mutations.
        self._index_dirty = set(range(len(self.pool.blocks)))
        self._index_nodes_dirty = set()
        self._protection_full = True
        self._index_hashes = [None] * len(self.pool.blocks)
        self._eligible_ids = set()
        self._protected_ids = set()
        self._resident = {}
        self._eligible_counts = Counter()
        self._index_parent = {}
        self._child_keys = defaultdict(set)
        self._blocked_children = Counter()
        self._blocked = set()
        self._reserved_last_copies = set()
        self._reachable = set()
        self._blank_eligible = set()
        self._candidate_groups = {}
        self._free_order = {block.block_id: rank for rank, block in enumerate(self.queue.iter_blocks_after(None))}
        self._order_low = 0
        self._order_high = len(self._free_order) - 1
        self.install()

    def _changed(self, reason):
        self.cache_epoch += 1
        self._available_cache = None
        self._structure_cache = None
        self.stats["invalidate_" + reason] += 1

    def current_cache_epoch(self):
        """O(1) version query; also detect whole-map replacement/reset."""
        for owner, name, sets in (
            (self.offload, "_block_id_to_pending_jobs", True),
            (self.pool, "_reuse_watchers", False),
        ):
            if owner is None or not hasattr(owner, name):
                continue
            current = getattr(owner, name)
            prior = self._tracked_mappings.get(name)
            if prior is None or current is not prior[2]:
                wrapped = _MutationDict(current, self._protection_changed, sets=sets)
                self._tracked_mappings[name] = (owner, current, wrapped)
                setattr(owner, name, wrapped)
                self._protection_full = True
                self._changed("protection_map")
        return self.cache_epoch

    def current_location_epoch(self):
        """Version of HBM hash presence, independent of physical eviction rules."""
        return self.location_epoch

    def _protection_changed(self, bid):
        if bid is None:
            self._protection_full = True
        else:
            self._index_dirty.add(bid)
        self._changed("protection")

    def _sync_index(self):
        """Apply changed native blocks, then propagate only affected ancestors.

        A content is blocked if any of its physical copies is ineligible or a
        resident child is blocked. If all of its own copies are eligible, exactly
        one of them must be retained; other copies remain legal victims.
        """
        self.current_cache_epoch()
        if self._protection_full:
            self._index_dirty.update(self._protected_ids)
            self._index_dirty.update(self._live_protected_block_ids())
            self._protection_full = False
        if not self._index_dirty and not self._index_nodes_dirty:
            return
        # Native cache_full_blocks normally observes append-only tails before
        # publication. Retain the fallback for direct native hash publishers.
        for rid, request in tuple(self.requests.items()):
            if rid in self.requests and rid not in self.finished and len(request.block_hashes) != len(self.chains[rid]):
                self.observe(request)
        pending = getattr(self.offload, "_block_id_to_pending_jobs", {})
        watchers = getattr(self.pool, "_reuse_watchers", {})
        nodes, self._index_nodes_dirty = self._index_nodes_dirty, set()
        # Newly observed ancestry can change a resident key's parent without
        # changing its physical hash. Keep it as an explicit topology seed.
        topology = set(nodes)
        presence_changed = set()
        dirty, self._index_dirty = self._index_dirty, set()
        self.stats["index_blocks_updated"] += len(dirty)
        self.stats["index_blocks_scanned"] += len(dirty)
        noops = 0
        for bid in dirty:
            block = self.pool.blocks[bid]
            protected = bool(pending.get(bid)) or bid in watchers
            if protected:
                self._protected_ids.add(bid)
            else:
                self._protected_ids.discard(bid)
            eligible = (
                not block.is_null
                and block.ref_cnt == 0
                and not protected
                and block.prev_free_block is not None
                and block.next_free_block is not None
            )
            was_eligible = bid in self._eligible_ids
            old_key = self._index_hashes[bid]
            key = block.block_hash if not block.is_null else None
            # Protection metadata and queue order still update immediately.
            # A fence change under a live reference, or a mutation reversed
            # before this sync, need not rebuild an unchanged physical index.
            if old_key == key and was_eligible == eligible:
                noops += 1
                continue
            if old_key == key:
                if key is not None:
                    self._eligible_counts[key] += 1 if eligible else -1
                    nodes.add(key)
            else:
                if old_key is not None:
                    if was_eligible:
                        self._eligible_counts[old_key] -= 1
                    copies = self._resident[old_key]
                    del copies[bid]
                    if not copies:
                        del self._resident[old_key]
                        if old_key in presence_changed:
                            presence_changed.remove(old_key)
                        else:
                            presence_changed.add(old_key)
                    nodes.add(old_key)
                if key is not None:
                    copies = self._resident.get(key)
                    if copies is None:
                        copies = self._resident[key] = {}
                        if key in presence_changed:
                            presence_changed.remove(key)
                        else:
                            presence_changed.add(key)
                    copies[bid] = block
                    if eligible:
                        self._eligible_counts[key] += 1
                    nodes.add(key)
            if eligible:
                self._eligible_ids.add(bid)
            else:
                self._eligible_ids.discard(bid)
            if eligible and key is None:
                self._blank_eligible.add(bid)
            else:
                self._blank_eligible.discard(bid)
            self._index_hashes[bid] = key
        self.stats["index_blocks_noop"] += noops
        self.stats["index_blocks_changed"] += len(dirty) - noops
        # Reachability depends on content presence, not physical-copy count,
        # refs or fences. Compare the whole batch's before/after presence so a
        # same-content copy replacement does not spuriously disconnect a chain.
        topology.update(presence_changed)
        # Direct parent links are present only for resident children, including
        # children whose parent is missing. Ref/fence/copy-count-only changes
        # cannot alter those edges and do not need to revisit them.
        self.stats["index_topology_checks"] += len(topology)
        for key in topology:
            existed = key in self._index_parent
            old_parent = self._index_parent.get(key)
            resident = key in self._resident
            parent = self.parents.get(key) if resident else None
            if existed and (not resident or old_parent != parent):
                if old_parent is not None:
                    self._child_keys[old_parent].discard(key)
                    if not self._child_keys[old_parent]:
                        del self._child_keys[old_parent]
                    if key in self._blocked:
                        self._blocked_children[old_parent] -= 1
                    nodes.add(old_parent)
                del self._index_parent[key]
            if resident and key not in self._index_parent:
                self._index_parent[key] = parent
                if parent is not None:
                    self._child_keys[parent].add(key)
                    if key in self._blocked:
                        self._blocked_children[parent] += 1
                    nodes.add(parent)
        # A draining set can remain sparse while parent updates keep adding one
        # key at a time. Keep deduplication separate from the FIFO worklist so
        # retrieving the next key never scans that sparse hash table.
        queued = set(nodes)
        work = deque(queued) if queued else ()
        ancestor_updates = 0
        while work:
            key = work.popleft()
            queued.remove(key)
            copies = self._resident.get(key, ())
            blocked = bool(copies) and (self._eligible_counts[key] < len(copies) or self._blocked_children[key] > 0)
            was_blocked = key in self._blocked
            if blocked:
                self._blocked.add(key)
            else:
                self._blocked.discard(key)
            if blocked and self._eligible_counts[key] == len(copies):
                self._reserved_last_copies.add(key)
            else:
                self._reserved_last_copies.discard(key)
            if blocked != was_blocked:
                parent = self._index_parent.get(key)
                if parent is not None:
                    self._blocked_children[parent] += 1 if blocked else -1
                    if parent not in queued:
                        queued.add(parent)
                        work.append(parent)
            ancestor_updates += 1
        self.stats["index_ancestor_updates"] += ancestor_updates
        # Reachability matters only for oracle ranking. A missing intermediate
        # prefix makes an otherwise resident suffix an orphan, just as before.
        self.stats["index_reachability_seeds"] += len(topology)
        queued = topology
        work = deque(queued) if queued else ()
        reachability_updates = 0
        while work:
            key = work.popleft()
            queued.remove(key)
            parent = self._index_parent.get(key)
            reachable = key in self._resident and (parent is None or parent in self._reachable)
            if reachable != (key in self._reachable):
                if reachable:
                    self._reachable.add(key)
                else:
                    self._reachable.discard(key)
                for child in self._child_keys.get(key, ()):
                    if child not in queued:
                        queued.add(child)
                        work.append(child)
            reachability_updates += 1
        self.stats["index_reachability_updates"] += reachability_updates
        # Drop transient zero counters for absent content; resident child counts
        # remain even when a parent is absent, so restoring it reconnects safely.
        for key in nodes:
            if key not in self._resident and self._eligible_counts[key] == 0:
                self._eligible_counts.pop(key, None)
            if self._blocked_children[key] == 0:
                self._blocked_children.pop(key, None)
            copies = self._resident.get(key, {})
            candidates = (
                {bid for bid in copies if bid in self._eligible_ids}
                if len(copies) > 1 or not self._child_keys.get(key)
                else set()
            )
            if candidates:
                self._candidate_groups[key] = candidates
            else:
                self._candidate_groups.pop(key, None)

    def observe(self, request):
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

        rid = request.request_id
        self._prune_consumed_owners()
        self.requests[rid] = request
        # The frozen Qwen setting has one full-attention group; reject silently
        # using group zero in a different layout.
        groups = self.scheduler.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError("oracle E currently requires the frozen single KV group")
        size = self.pool.hash_block_size
        actual = groups[0].kv_cache_spec.block_size
        if actual != size:
            raise ValueError("oracle E requires the frozen matching hash/block size")
        chain = self.chains.setdefault(rid, [])
        if len(request.block_hashes) < len(chain):
            raise ValueError("observed request hashes must be append-only")
        ancestry_changed = False
        for block_hash in request.block_hashes[len(chain) :]:
            key = make_block_hash_with_group_id(block_hash, 0)
            parent = chain[-1] if chain else None
            if key in self.parents and self.parents[key] != parent:
                raise ValueError("same native content key has conflicting parent")
            ancestry_changed |= key not in self.parents
            if key not in self.parents:
                self._index_nodes_dirty.add(key)
            self.parents[key], self.depth[key] = parent, len(chain)
            self.content_owners[key].add(rid)
            chain.append(key)
        if ancestry_changed:
            self._changed("ancestry")

    def on_complete(self, request, advice):
        self.observe(request)
        self.finished.add(request.request_id)
        deadline = self._deadline(request)
        if deadline is None:
            self.stats["expected_unknown_owners"] += 1
        if deadline == math.inf:
            self._retire_owner(request.request_id)

    def _live_protected_block_ids(self):
        pending = getattr(self.offload, "_block_id_to_pending_jobs", {})
        protected = {bid for bid, jobs in pending.items() if jobs}
        # Native unpin watchers may protect a source until safe reuse even if
        # it is absent from the connector's normal pending-job map.
        protected.update(getattr(self.pool, "_reuse_watchers", {}))
        return protected

    def protected_block_ids(self):
        self._sync_index()
        return set(self._protected_ids)

    def _deadline(self, request):
        if request.request_id not in self.finished:
            return float(self.state.clock.monotonic())
        advice = self.state.advice_for_request(request.request_id)
        status = getattr(advice, "status", None)
        if status == "unknown":
            expected = getattr(self.state, "is_expected_unknown", lambda advice: False)(advice)
            if not expected:
                if hasattr(self.state, "fail"):
                    self.state.fail("E received unexpected unknown advice", request_id=request.request_id)
                raise ValueError("oracle E requires a whitelisted unknown")
            if self.state.is_advice_consumed(advice):
                return math.inf  # consumed demand, not an unknown-time sentinel
            return None
        if status == "no_direct_successor":
            return math.inf
        if status == "resolved":
            successors = getattr(advice, "successors", ())
            remaining = [s for s in successors if s.request_id not in self.state.arrivals]
            if successors and not remaining:
                return math.inf
            if not successors and getattr(advice, "successor_request_id", None) in self.state.arrivals:
                return math.inf
            value = min((s.return_time for s in remaining), default=getattr(advice, "return_time", None))
            if value is None or not math.isfinite(value):
                raise ValueError("resolved E advice must have a finite arrival")
            return float(value)
        raise ValueError(f"oracle E cannot use advice status {status!r}")

    def _retire_owner(self, rid):
        """Drop consumed demand; content ancestry remains for other/current owners."""
        for key in self.chains.pop(rid, ()):
            owners = self.content_owners[key]
            owners.discard(rid)
            if not owners:
                del self.content_owners[key]
        self.requests.pop(rid, None)
        self.finished.discard(rid)
        self.stats["retired_owners"] += 1

    def _prune_consumed_owners(self):
        # Only currently live owners are checked, never the historical request set.
        for rid in tuple(self.finished):
            request = self.requests.get(rid)
            if request is not None and self._deadline(request) == math.inf:
                self._retire_owner(rid)

    def _retained_contents(self, resident, eligible_counts):
        """Contents whose last copy cannot be legally removed in this preview."""
        blocked = {key for key, copies in resident.items() if eligible_counts[key] < len(copies)}
        pending = list(blocked)
        while pending:
            parent = self.parents.get(pending.pop())
            if parent is not None and parent in resident and parent not in blocked:
                blocked.add(parent)
                pending.append(parent)
        return blocked

    def _scan_structure(self):
        """Slow physical reference/rebuild; never called for an unchanged query."""
        # Native decoding appends hashes between admission and completion. Observe
        # only new tails, including prefixes about to become free on preemption.
        for rid, request in tuple(self.requests.items()):
            # observe() may retire another consumed owner in this tuple.
            if rid in self.requests and rid not in self.finished and len(request.block_hashes) != len(self.chains[rid]):
                self.observe(request)
        protected = self._live_protected_block_ids()
        free = list(self.queue.iter_blocks_after(None))
        resident = defaultdict(list)
        for block in self.pool.blocks:
            if block.block_hash is not None and not block.is_null:
                resident[block.block_hash].append(block)
        children = Counter()
        for key in resident:
            parent = self.parents.get(key)
            if parent is not None and parent in resident:
                children[parent] += 1
        self.stats["structure_rebuilds"] += 1
        self.stats["physical_slots_scanned"] += len(self.pool.blocks)
        self.stats["free_entries_scanned"] += len(free)
        return free, resident, children, protected

    def _structure(self, protected_ids=()):
        epoch = self.current_cache_epoch()
        if self._structure_cache is None or self._structure_cache[0] != epoch:
            value = self._scan_structure()
            # Registering an appended observed tail can advance the epoch.
            self._structure_cache = (self.cache_epoch, value)
        else:
            self.stats["structure_cache_hits"] += 1
        free, resident, children, protected = self._structure_cache[1]
        if protected_ids:
            protected = protected | set(protected_ids)
        return free, resident, children, protected

    def _count_available(self, free, resident, protected):
        eligible = [b for b in free if not b.is_null and b.ref_cnt == 0 and b.block_id not in protected]
        by_content = Counter(b.block_hash for b in eligible if b.block_hash is not None)
        blocked = self._retained_contents(resident, by_content)
        retained_last_copies = sum(by_content[key] == len(resident[key]) for key in blocked)
        return len(eligible) - retained_last_copies, retained_last_copies

    def reference_available(self):
        """Slow audit oracle for tests; does not validate cached state on hot paths."""
        self.current_cache_epoch()
        free, resident, _, protected = self._scan_structure()
        return self._count_available(free, resident, protected)[0]

    def available(self):
        """Count all legally peelable physical blocks, including duplicate copies.

        A non-evictable descendant forces every resident ancestor content to stay.
        Such an ancestor may still lose duplicate copies, but never its last copy.
        This is the exact cardinality of tail peeling, not a pending-only bound.
        """
        start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            self._sync_index()
            epoch = self.cache_epoch
            if self._available_cache is not None and self._available_cache[0] == epoch:
                self.stats["availability_cache_hits"] += 1
                return self._available_cache[1]
            retained_last_copies = len(self._reserved_last_copies)
            result = len(self._eligible_ids) - retained_last_copies
            self.stats["availability_checks"] += 1
            self.stats["ancestor_last_copy_reservations"] += retained_last_copies
            self._available_cache = (self.cache_epoch, result)
            return result
        finally:
            self.seconds += time.perf_counter() - start
            self.thread_cpu_seconds += time.thread_time() - cpu_start

    def _snapshot(self, protected_ids):
        self._prune_consumed_owners()
        free, resident, children, protected = self._scan_structure()
        protected |= set(protected_ids)
        # Appended decode hashes are registered as native lookup/complete hooks see
        # them. A snapshot never walks every request's full historical prefix.
        owner_deadlines = {rid: self._deadline(request) for rid, request in self.requests.items()}
        self.stats["owner_deadlines_examined"] += len(owner_deadlines)
        reachable = {}

        def prefix_is_resident(key):
            trail = []
            current = key
            while current is not None and current not in reachable:
                if current not in resident:
                    value = False
                    break
                trail.append(current)
                current = self.parents.get(current)
            else:
                value = True if current is None else reachable[current]
            for item in trail:
                reachable[item] = value
            return value

        deadlines, unknown_contents = {}, set()
        for key in resident:
            owners = self.content_owners.get(key, ())
            if any(owner_deadlines[rid] is None for rid in owners):
                # One unknown owner makes the shared earliest deadline unknown;
                # never silently replace it with min(the known owners).
                unknown_contents.add(key)
                continue
            if not prefix_is_resident(key):
                continue
            if owners:
                deadlines[key] = min(owner_deadlines[rid] for rid in owners)
        self.stats["resident_contents_examined"] += len(resident)
        return free, resident, children, deadlines, protected, unknown_contents

    def preview(self, count, protected_ids=()):
        """Return exact eligible victims without changing native state."""
        return self._preview(count, self._checked_protected_ids(protected_ids), allow_partial=False)

    def preview_up_to(self, count, protected_ids=()):
        """Return the longest legal selection up to count, for prefix shortening."""
        return self._preview(count, self._checked_protected_ids(protected_ids), allow_partial=True)

    def _checked_protected_ids(self, protected_ids):
        protected = set(protected_ids)
        if any(type(bid) is not int or not 0 <= bid < len(self.pool.blocks) for bid in protected):
            raise ValueError("protected block ID must be an integer in the native pool")
        return protected

    def reference_preview(self, count, protected_ids=(), *, allow_partial=False):
        """Independent old full-scan algorithm, only for differential audits."""
        protected_ids = self._checked_protected_ids(protected_ids)
        previous = self._last_preview
        try:
            return self._reference_preview(count, protected_ids, allow_partial=allow_partial)
        finally:
            self._last_preview = previous

    def _preview(self, count, protected_ids, *, allow_partial):
        if count < 0:
            raise ValueError("negative allocation")
        if count == 0:
            self._last_preview = ((), "blank", 0)
            return []
        start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            self._sync_index()
            self._prune_consumed_owners()
            extra = set(protected_ids)
            protected = self._protected_ids | extra
            blank = []
            for block in self.queue.iter_blocks_after(None):
                if block.block_hash is not None:
                    break
                if block.block_id not in protected and block.ref_cnt == 0 and not block.is_null:
                    blank.append(block)
                    if len(blank) == count:
                        self.stats["blank_selection_fast_paths"] += 1
                        self._last_preview = (tuple(b.block_id for b in blank), "blank", 0)
                        return blank
            # Native queue operations only remove, prepend, or append. Monotone
            # integer order labels preserve its exact LRU ties without walking
            # all free entries. Commit still verifies the real queue links.
            rank = self._free_order
            blocked = self._blocked
            if extra:
                blocked = set(blocked)
                for bid in extra:
                    key = self.pool.blocks[bid].block_hash
                    while key is not None and key in self._resident and key not in blocked:
                        blocked.add(key)
                        key = self.parents.get(key)
            owner_deadlines = {rid: self._deadline(request) for rid, request in self.requests.items()}
            self.stats["owner_deadlines_examined"] += len(owner_deadlines)
            unknown_candidates = {
                key
                for rid, deadline in owner_deadlines.items()
                if deadline is None
                for key in self.chains[rid]
                if key in self._resident and key not in blocked
            }
            use_lru = bool(unknown_candidates)
            source = "history_lru" if use_lru else "oracle"
            self.stats["lru_fallback_previews" if use_lru else "oracle_ranked_previews"] += 1
            self.stats["unknown_candidate_contents_examined"] += len(unknown_candidates)
            removed_copies, removed_children = Counter(), Counter()
            selected, chosen, heap, deadlines = [], set(), [], {}

            def copies_left(key):
                return len(self._resident[key]) - removed_copies[key]

            def children_left(key):
                return len(self._child_keys.get(key, ())) - removed_children[key]

            def priority(block):
                key, bid = block.block_hash, block.block_id
                if key is None:
                    return (0, 0.0, rank[bid], bid)
                if copies_left(key) > 1:
                    return (1, 0.0, rank[bid], bid)
                if use_lru:
                    urgency = 0.0
                else:
                    if key not in deadlines:
                        owners = self.content_owners.get(key, ())
                        deadlines[key] = (
                            min(owner_deadlines[rid] for rid in owners)
                            if owners and key in self._reachable
                            else math.inf
                        )
                        self.stats["candidate_deadlines_evaluated"] += 1
                    urgency = -deadlines[key]
                return (2, urgency, rank[bid], bid)

            def push(block):
                if block.block_id not in self._eligible_ids or block.block_id in extra or block.block_id in chosen:
                    return
                key = block.block_hash
                if key is not None and children_left(key) and copies_left(key) == 1:
                    return
                heapq.heappush(heap, (priority(block), block.block_id))

            for bid in self._blank_eligible:
                push(self.pool.blocks[bid])
            self.stats["preview_initial_candidates"] += len(self._blank_eligible)
            for candidates in self._candidate_groups.values():
                for bid in candidates:
                    push(self.pool.blocks[bid])
                self.stats["preview_initial_candidates"] += len(candidates)
            while len(selected) < count:
                if not heap:
                    if allow_partial:
                        break
                    raise InsufficientEvictableBlocks(f"need {count}, only {len(selected)} safe tail blocks")
                old, bid = heapq.heappop(heap)
                if bid in chosen:
                    continue
                block = self.pool.blocks[bid]
                key = block.block_hash
                if key is not None and children_left(key) and copies_left(key) == 1:
                    continue
                if old != priority(block):
                    push(block)
                    continue
                chosen.add(bid)
                selected.append(block)
                if key is not None:
                    removed_copies[key] += 1
                    if copies_left(key) == 0:
                        parent = self.parents.get(key)
                        if parent is not None:
                            removed_children[parent] += 1
                            if children_left(parent) == 0:
                                for candidate in self._resident.get(parent, {}).values():
                                    push(candidate)
                    else:
                        for candidate in self._resident[key].values():
                            push(candidate)
            self._last_preview = (tuple(b.block_id for b in selected), source, len(unknown_candidates))
            return selected
        finally:
            self.seconds += time.perf_counter() - start
            self.thread_cpu_seconds += time.thread_time() - cpu_start

    def _reference_preview(self, count, protected_ids, *, allow_partial):
        if count < 0:
            raise ValueError("negative allocation")
        if count == 0:
            self._last_preview = ((), "blank", 0)
            return []
        start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            self.current_cache_epoch()
            protected_now = self._live_protected_block_ids() | set(protected_ids)
            blank = []
            for block in self.queue.iter_blocks_after(None):
                if block.block_hash is not None:
                    break
                if block.block_id not in protected_now and block.ref_cnt == 0 and not block.is_null:
                    blank.append(block)
                    if len(blank) == count:
                        self.stats["blank_selection_fast_paths"] += 1
                        self._last_preview = (tuple(b.block_id for b in blank), "blank", 0)
                        return blank
            free, resident, children, deadlines, protected, unknown_contents = self._snapshot(protected_ids)
            # Selection peels a private working copy, never the reusable snapshot.
            resident = {key: list(copies) for key, copies in resident.items()}
            children = children.copy()
            by_id = {b.block_id: b for b in free if b.block_id not in protected and not b.is_null and b.ref_cnt == 0}
            rank = {b.block_id: i for i, b in enumerate(free)}
            eligible_counts = Counter(b.block_hash for b in by_id.values() if b.block_hash is not None)
            unknown_candidates = unknown_contents - self._retained_contents(resident, eligible_counts)
            use_lru = bool(unknown_candidates)
            source = "history_lru" if use_lru else "oracle"
            self.stats["lru_fallback_previews" if use_lru else "oracle_ranked_previews"] += 1
            self.stats["unknown_candidate_contents_examined"] += len(unknown_candidates)
            selected, chosen, heap = [], set(), []

            def priority(block):
                key = block.block_hash
                if key is None:
                    return (0, 0.0, rank[block.block_id], block.block_id)
                # Freeing a duplicate does not erase the content.
                if len(resident[key]) > 1:
                    return (1, 0.0, rank[block.block_id], block.block_id)
                urgency = 0.0 if use_lru else -deadlines.get(key, math.inf)
                return (2, urgency, rank[block.block_id], block.block_id)

            def push(block):
                if block.block_id not in by_id or block.block_id in chosen:
                    return
                key = block.block_hash
                if key is not None and children[key] and len(resident[key]) == 1:
                    return
                heapq.heappush(heap, (priority(block), block.block_id))

            for block in by_id.values():
                push(block)
            while len(selected) < count:
                if not heap:
                    if allow_partial:
                        break
                    raise InsufficientEvictableBlocks(f"need {count}, only {len(selected)} safe tail blocks")
                old, bid = heapq.heappop(heap)
                if bid in chosen:
                    continue
                block = by_id[bid]
                key = block.block_hash
                if key is not None and children[key] and len(resident[key]) == 1:
                    continue
                if old != priority(block):
                    push(block)
                    continue
                chosen.add(bid)
                selected.append(block)
                if key is not None:
                    resident[key].remove(block)
                    if not resident[key]:
                        parent = self.parents.get(key)
                        if parent is not None:
                            children[parent] -= 1
                            if children[parent] == 0:
                                for candidate in resident.get(parent, ()):
                                    push(candidate)
                    else:
                        for candidate in resident[key]:
                            push(candidate)
            self._last_preview = (tuple(b.block_id for b in selected), source, len(unknown_candidates))
            return selected
        finally:
            self.seconds += time.perf_counter() - start
            self.thread_cpu_seconds += time.thread_time() - cpu_start

    @contextlib.contextmanager
    def reserve_selection(self, blocks):
        """Require exact victims; yield cancellation for an untouched denial."""
        if self._forced is not None:
            raise RuntimeError("nested eviction reservation")
        expected = [(b, b.block_hash) for b in blocks]
        original = tuple(expected)
        self._forced = expected
        bids = tuple(b.block_id for b in blocks)
        if bids == self._last_preview[0][: len(bids)]:
            self._forced_info = self._last_preview[1:]
            if all(block.block_hash is None for block in blocks):
                self._forced_info = ("blank", 0)
        else:
            self._forced_info = ("explicit_reservation", 0)

        def cancel_unconsumed():
            # A legal native admission rejection returns None before touching
            # the queue. Never hide a partially committed allocation or a
            # changed victim behind a cancellation.
            if self._forced is not expected or tuple(expected) != original:
                raise RuntimeError("cannot cancel a consumed eviction reservation")
            if any(block.block_hash != key or block.ref_cnt for block, key in expected):
                raise RuntimeError("cannot cancel a changed eviction reservation")
            expected.clear()
            self.stats["cancelled_unconsumed_reservations"] += 1
            self.emit(
                {
                    "kind": "E_reservation_cancelled",
                    "time": self.state.clock.monotonic(),
                    "reason": "native_admission_denied_before_allocation",
                    "count": len(original),
                }
            )

        try:
            yield cancel_unconsumed
            if self._forced:
                raise RuntimeError("reserved eviction victims were not consumed")
        finally:
            self._forced = None
            self._forced_info = None

    def _pop(self, count):
        if self._forced is not None:
            selection_source, unknown_count = self._forced_info
            if count > len(self._forced):
                raise ValueError("native allocation exceeds preview")
            pairs = self._forced[:count]
            del self._forced[:count]
            blocks = [b for b, key in pairs]
            protected = self.protected_block_ids()
            for block, key in pairs:
                if block.block_hash != key or block.ref_cnt or block.block_id in protected:
                    raise ValueError("previewed block changed before native allocation")
        else:
            blocks = self.preview(count)
            selection_source, unknown_count = self._last_preview[1:]
        for block in blocks:
            if block.prev_free_block is None or block.next_free_block is None:
                raise ValueError("chosen victim no longer belongs to native free queue")
        self.emit(
            {
                "kind": "E_allocate",
                "time": self.state.clock.monotonic(),
                "count": count,
                "block_ids": [b.block_id for b in blocks],
                "cached_victims": sum(b.block_hash is not None for b in blocks),
                "selection_source": selection_source,
                "unknown_candidate_contents": unknown_count,
            }
        )
        for block in blocks:
            self.queue.remove(block)
        self.stats["allocations"] += 1
        self.stats[selection_source + "_allocations"] += 1
        self.stats[selection_source + "_allocated_blocks"] += count
        self.stats["blocks"] += count
        return blocks

    def install(self):
        # Queue changes cover native allocation and ref_cnt crossings of zero in
        # touch/free/unpin. Nonzero-to-nonzero refs cannot change legal capacity.
        for name in ("popleft", "remove", "append", "prepend_n", "append_n"):
            original = getattr(self.queue, name)

            def mutate(*args, _original=original, _name=name, **kwargs):
                before = self.queue.num_free_blocks
                result = None
                try:
                    result = _original(*args, **kwargs)
                    return result
                finally:
                    if self.queue.num_free_blocks != before:
                        if _name == "popleft":
                            blocks = [result]
                        else:
                            plural = _name in {"prepend_n", "append_n"}
                            value = args[0] if args else kwargs["blocks" if plural else "block"]
                            blocks = value if plural else [value]
                        if _name in {"remove", "popleft"}:
                            for block in blocks:
                                self._free_order.pop(block.block_id)
                        elif _name == "prepend_n":
                            self._order_low -= len(blocks)
                            for offset, block in enumerate(blocks):
                                self._free_order[block.block_id] = self._order_low + offset
                        else:
                            for block in blocks:
                                self._order_high += 1
                                self._free_order[block.block_id] = self._order_high
                        self._index_dirty.update(block.block_id for block in blocks)
                        self._changed("free_queue")

            self._mutation_hooks.append((self.queue, name, original))
            setattr(self.queue, name, mutate)

        original_insert = self.pool._insert_block_hash

        def insert(block_hash_with_group_id, block, num_tokens=None):
            present = self.pool.cached_block_hash_to_block.get_one_block(block_hash_with_group_id) is not None
            before = self.pool.cached_block_hash_to_block.contain(block_hash_with_group_id, block.block_id)
            result = original_insert(block_hash_with_group_id, block, num_tokens=num_tokens)
            if not before and self.pool.cached_block_hash_to_block.contain(block_hash_with_group_id, block.block_id):
                self._index_dirty.add(block.block_id)
                self._changed("hash")
                if not present:
                    self.location_epoch += 1
                    self.stats["invalidate_location"] += 1
            return result

        original_remove = self.pool._remove_cached_block_hashes

        def remove(block):
            before = block.block_hash
            result = original_remove(block)
            if result or block.block_hash != before:
                self._index_dirty.add(block.block_id)
                self._changed("hash")
                # Native blocks may also own alias hashes. Removing an alias's
                # last copy changes lookup even if the primary still has a copy.
                if any(self.pool.cached_block_hash_to_block.get_one_block(key) is None for key in result):
                    self.location_epoch += 1
                    self.stats["invalidate_location"] += 1
            return result

        original_reset = self.pool.reset_prefix_cache

        def reset():
            result = original_reset()
            if result:
                self._index_dirty.update(range(len(self.pool.blocks)))
                self._changed("reset")
                self.location_epoch += 1
                self.stats["invalidate_location"] += 1
            return result

        original_cache = self.pool.cache_full_blocks

        def cache(request, *args, **kwargs):
            rid = request.request_id
            if rid in self.requests and len(request.block_hashes) != len(self.chains[rid]):
                self.observe(request)
            return original_cache(request, *args, **kwargs)

        for name, original, wrapped in (
            ("_insert_block_hash", original_insert, insert),
            ("_remove_cached_block_hashes", original_remove, remove),
            ("reset_prefix_cache", original_reset, reset),
            ("cache_full_blocks", original_cache, cache),
        ):
            self._mutation_hooks.append((self.pool, name, original))
            setattr(self.pool, name, wrapped)

        def lookup(request, *args, **kwargs):
            self.observe(request)
            return self._original_lookup(request, *args, **kwargs)

        def allocate(count):
            result = self._original_allocate(count)
            self.on_cache_change()
            return result

        self.queue.popleft_n = self._pop
        self.pool.get_num_free_blocks = self.available
        self.pool.get_new_blocks = allocate
        self.scheduler.kv_cache_manager.get_computed_blocks = lookup
        self.state.subscribe(self.on_complete)
        self.current_cache_epoch()
        self.pool._joint_cache_epoch = self.current_cache_epoch
        self.pool._joint_location_epoch = self.current_location_epoch

    def close(self):
        for owner, name, original in reversed(self._mutation_hooks):
            setattr(owner, name, original)
        self._mutation_hooks.clear()
        for name, (owner, original, wrapped) in self._tracked_mappings.items():
            if getattr(owner, name) is wrapped:
                original.clear()
                original.update({key: set(value) if wrapped.sets else value for key, value in wrapped.items()})
                setattr(owner, name, original)
        self._tracked_mappings.clear()
        if self._previous_epoch_hook is None:
            del self.pool._joint_cache_epoch
        else:
            self.pool._joint_cache_epoch = self._previous_epoch_hook
        if self._previous_location_epoch_hook is None:
            del self.pool._joint_location_epoch
        else:
            self.pool._joint_location_epoch = self._previous_location_epoch_hook
        self.queue.popleft_n = self._original_pop
        self.pool.get_num_free_blocks = self._original_free_count
        self.pool.get_new_blocks = self._original_allocate
        self.scheduler.kv_cache_manager.get_computed_blocks = self._original_lookup
