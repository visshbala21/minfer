"""Radix-tree prefix cache: token prefixes mapped to shared, reference-counted KV blocks.

Agent workloads resend the same tokens over and over: every request of an agent type starts
with the same system prompt and tool schemas, and every turn of a session starts with the
whole previous turn. The keys and values of a token depend only on the tokens up to it, so
they can be computed once and shared by every request with the same prefix.

The tree is keyed on token sequences. Each node owns one edge: a run of tokens and the KV
blocks holding them. Edges are *block aligned* — an edge is a whole number of blocks —
because a block is the unit the page tables can share; a prefix is therefore reused up to
its last full block, and at most ``block_size - 1`` matching tokens are recomputed.
Children are indexed by the tokens of their first block, so descending the tree costs one
dictionary lookup per node plus a comparison along the edge.

The cache holds one reference to every block in the tree (see
:meth:`BlockManager.cache_block`); sequences that use a cached prefix hold their own
references through their block tables. A block referenced only by the cache is evictable.
Eviction removes least-recently-used leaves, from the tail of the edge backwards — the
tokens furthest from the root are the least likely to be shared.

**Host tier.** With a :class:`~minfer.cache.host_tier.HostTier`, a node's blocks can also
have copies in pinned host memory (``RadixNode.host``). Evicting a block that has one does
not forget its tokens: the trimmed tail becomes a *host-only* node (no GPU blocks, same
tokens), which a later request can have copied back instead of recomputing
(:meth:`PrefixCache.host_continuation`, :meth:`PrefixCache.begin_restore`). A host-only
node is invisible to :meth:`PrefixCache.sync` and :meth:`PrefixCache.match_length`, which
describe what is usable on the GPU right now.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass
from itertools import count

from minfer.cache.block_manager import BlockManager
from minfer.cache.host_tier import HostTier, Transfer
from minfer.cache.retention import SessionRetention, SessionState


class RadixNode:
    """A tree node and the edge leading to it.

    Attributes:
        tokens: The edge's tokens; ``len(tokens) == len(blocks) * block_size``.
        blocks: KV blocks holding the edge's tokens, in position order.
        children: Child nodes, keyed by the tokens of their edge's first block.
        parent: The parent node; ``None`` for the root and for nodes removed from the tree.
        last_access: Logical time the node was last matched or inserted.
        last_used: Wall-clock time of the same event (for session-aware retention).
        owner: The session whose request last finished on this node, if any.
        host: Host-tier slots holding copies of the edge's leading blocks (possibly all of
            them, possibly none). A node with tokens but no ``blocks`` is *host-only*.
        restoring: A copy from the host into new GPU blocks is in flight for this node.
    """

    __slots__ = (
        "blocks",
        "children",
        "host",
        "last_access",
        "last_used",
        "owner",
        "parent",
        "restoring",
        "tokens",
    )

    def __init__(self, tokens: tuple[int, ...], blocks: list[int], parent: RadixNode | None):
        self.tokens = tokens
        self.blocks = blocks
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.parent = parent
        self.last_access = 0
        self.last_used = 0.0
        self.owner: SessionState | None = None
        self.host: list[int] = []
        self.restoring = False


@dataclass
class RestoreTicket:
    """Host-only nodes being copied back to the GPU.

    Attributes:
        nodes: The nodes, in path order, with the GPU blocks each one will get.
        blocks: All of those blocks; the ticket holds one reference to each until
            :meth:`PrefixCache.release_restore`.
        transfer: The copy; the nodes become GPU-resident once it is done and
            :meth:`PrefixCache.finish_restore` has run.
        finished: Whether :meth:`PrefixCache.finish_restore` has run.
    """

    nodes: list[tuple[RadixNode, list[int]]]
    blocks: list[int]
    transfer: Transfer
    finished: bool = False


@dataclass
class PrefixCacheStats:
    """Monotonic counters of prefix-cache activity.

    ``lookup_tokens``/``hit_tokens`` are counted when a sequence is admitted: how many
    tokens it needed in the cache, and how many it found there. ``catchup_tokens`` are
    tokens a sequence adopted later, while still prefilling, because another sequence
    computed them first. Both kinds are prefill work that was not done.
    """

    lookups: int = 0
    hits: int = 0
    lookup_tokens: int = 0
    hit_tokens: int = 0
    catchup_tokens: int = 0
    inserted_blocks: int = 0
    evicted_blocks: int = 0
    demoted_blocks: int = 0
    restored_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of looked-up tokens served from the cache."""
        return self.hit_tokens / self.lookup_tokens if self.lookup_tokens else 0.0

    @property
    def prefill_tokens_saved(self) -> int:
        """Tokens whose KV was reused instead of computed."""
        return self.hit_tokens + self.catchup_tokens


class PrefixCache:
    """Radix tree over token sequences whose nodes hold shared KV blocks.

    Args:
        blocks: The block pool the cached KV lives in.
        retention: Session-aware eviction order (:mod:`minfer.cache.retention`). Without
            it, eviction is least-recently-used.
        clock: Monotonic time source, used only with ``retention``.
        host: Host-memory tier (needs ``retention``): evicted KV that has a host copy stays
            in the tree as host-only nodes.
    """

    def __init__(
        self,
        blocks: BlockManager,
        retention: SessionRetention | None = None,
        clock: Callable[[], float] = time.monotonic,
        host: HostTier | None = None,
    ) -> None:
        if host is not None and retention is None:
            raise ValueError("the host tier needs the session-aware retention policy")
        self.blocks = blocks
        self.block_size = blocks.block_size
        self.retention = retention
        self.clock = clock
        self.host = host
        self.root = RadixNode((), [], None)
        self.num_nodes = 0
        self.num_blocks = 0
        self.stats = PrefixCacheStats()
        self._clock = 0
        self._now = 0.0
        # Min-heap of (last_access, tiebreak, node) with lazy invalidation: an entry is
        # current only if the node is still a leaf of the tree with that access time.
        self._heap: list[tuple[int, int, RadixNode]] = []
        self._tiebreak = count()

    # ------------------------------------------------------------------ walk

    def _matching_blocks(
        self, node: RadixNode, tokens: Sequence[int], start: int, limit: int
    ) -> int:
        """Leading blocks of ``node``'s edge equal to ``tokens`` from block ``start``.

        At most ``limit`` blocks are compared. The first block is known to match (it is the
        key the node was found under).
        """
        size = self.block_size
        span = min(len(node.tokens) // size, limit)
        begin = start * size
        if tuple(tokens[begin : begin + span * size]) == node.tokens[: span * size]:
            return span
        edge = node.tokens
        matched = 1
        while matched < span:
            lo = matched * size
            if tuple(tokens[begin + lo : begin + lo + size]) != edge[lo : lo + size]:
                break
            matched += 1
        return matched

    def match_length(self, tokens: Sequence[int]) -> int:
        """Tokens of ``tokens`` a new sequence could take from the cache right now.

        Read-only (does not count as a use for LRU). At least one token is always left
        uncached, because a sequence needs a pending token to produce logits from.
        """
        size = self.block_size
        limit = (len(tokens) - 1) // size
        node, index = self.root, 0
        while index < limit:
            child = node.children.get(tuple(tokens[index * size : (index + 1) * size]))
            if child is None or not child.blocks:
                break
            matched = self._matching_blocks(child, tokens, index, limit - index)
            index += matched
            if matched < len(child.blocks):
                break
            node = child
        return index * size

    def sync(self, tokens: Sequence[int], own_blocks: Sequence[int], limit: int) -> list[int]:
        """Reconcile a sequence's leading blocks with the tree, in both directions.

        Walks the tree along ``tokens``. Where the tree already holds a block for a
        position, that block is the canonical one; where it does not and the sequence has
        computed one (``own_blocks``), the sequence's block is inserted. The walk continues
        past the sequence's own blocks for as long as the tree has matching ones.

        Args:
            tokens: The sequence's tokens.
            own_blocks: Blocks the sequence has completely filled, in position order.
            limit: Maximum number of leading blocks to return.

        Returns:
            The canonical blocks of the longest prefix (≤ ``limit`` blocks) that is in the
            tree after the call. It covers at least ``own_blocks[:limit]``. The caller must
            point its table at these blocks (:meth:`BlockManager.adopt`).
        """
        size = self.block_size
        num_own = min(len(own_blocks), limit)
        self._clock += 1
        if self.retention is not None:
            self._now = self.clock()
        canonical: list[int] = []
        node, index = self.root, 0
        while index < limit:
            key = tuple(tokens[index * size : (index + 1) * size])
            child = node.children.get(key)
            if child is None:
                if index < num_own:
                    grow = self._extend_leaf if self._is_leaf(node) else self._add_child
                    canonical.extend(grow(node, tokens, own_blocks, index, num_own))
                break
            if not child.blocks:
                # Host-only: nothing here is usable on the GPU. If the sequence computed
                # these positions itself, its blocks bring the node back.
                if index >= num_own or child.restoring:
                    break
                matched = self._matching_blocks(child, tokens, index, num_own - index)
                self._promote(child, own_blocks[index : index + matched])
                canonical.extend(child.blocks)
                index += matched
                node = child
                continue
            matched = self._matching_blocks(child, tokens, index, limit - index)
            if matched < len(child.blocks) and index + matched < num_own:
                # The sequence diverges inside the edge and has blocks of its own to hang
                # below the divergence point.
                self._split(child, matched)
            self._touch(child)
            canonical.extend(child.blocks[:matched])
            index += matched
            if matched < len(child.blocks):
                break
            node = child
        return canonical

    def _is_leaf(self, node: RadixNode) -> bool:
        return node is not self.root and not node.children

    def _promote(self, node: RadixNode, own_blocks: Sequence[int]) -> None:
        """Give a host-only node GPU blocks a sequence computed for its leading tokens."""
        if len(own_blocks) * self.block_size < len(node.tokens):
            self._split(node, len(own_blocks))
        node.blocks = list(own_blocks)
        self._account_insert(node, node.blocks)

    def _add_child(
        self,
        parent: RadixNode,
        tokens: Sequence[int],
        own_blocks: Sequence[int],
        start: int,
        stop: int,
    ) -> list[int]:
        """Hang ``own_blocks[start:stop]`` below ``parent`` as a new leaf."""
        size = self.block_size
        new_blocks = list(own_blocks[start:stop])
        child = RadixNode(tuple(tokens[start * size : stop * size]), new_blocks, parent)
        parent.children[child.tokens[:size]] = child
        self.num_nodes += 1
        self._account_insert(child, new_blocks)
        return new_blocks

    def _extend_leaf(
        self,
        leaf: RadixNode,
        tokens: Sequence[int],
        own_blocks: Sequence[int],
        start: int,
        stop: int,
    ) -> list[int]:
        """Append ``own_blocks[start:stop]`` to a leaf's edge.

        A sequence that keeps generating would otherwise grow a chain of one-block nodes;
        lengthening the leaf keeps the depth of the tree proportional to the number of
        *divergences*, not to the number of tokens.
        """
        size = self.block_size
        new_blocks = list(own_blocks[start:stop])
        leaf.tokens += tuple(tokens[start * size : stop * size])
        leaf.blocks.extend(new_blocks)
        self._account_insert(leaf, new_blocks)
        return new_blocks

    def _account_insert(self, node: RadixNode, new_blocks: list[int]) -> None:
        for block in new_blocks:
            self.blocks.cache_block(block)
        self.num_blocks += len(new_blocks)
        self.stats.inserted_blocks += len(new_blocks)
        self._touch(node)

    def _split(self, node: RadixNode, num_blocks: int) -> None:
        """Cut ``node``'s edge after ``num_blocks`` blocks.

        ``node`` keeps the head of the edge (and its place under its parent); a new node
        takes the tail and all of ``node``'s children.
        """
        cut = num_blocks * self.block_size
        tail = RadixNode(node.tokens[cut:], node.blocks[num_blocks:], node)
        tail.children = node.children
        for grandchild in tail.children.values():
            grandchild.parent = tail
        tail.last_access = node.last_access
        tail.last_used = node.last_used
        tail.owner = node.owner
        tail.host = node.host[num_blocks:]
        node.host = node.host[:num_blocks]
        node.tokens = node.tokens[:cut]
        node.blocks = node.blocks[:num_blocks]
        node.children = {tail.tokens[: self.block_size]: tail}
        self.num_nodes += 1
        if not tail.children and tail.blocks:
            heapq.heappush(self._heap, (tail.last_access, next(self._tiebreak), tail))

    def _touch(self, node: RadixNode) -> None:
        """Record a use of ``node``."""
        node.last_access = self._clock
        node.last_used = self._now
        if not node.children:
            heapq.heappush(self._heap, (self._clock, next(self._tiebreak), node))
            if len(self._heap) > 64 + 8 * self.num_nodes:
                self._rebuild_heap()

    def _rebuild_heap(self) -> None:
        """Drop stale heap entries (every touch of a leaf leaves its older entry behind)."""
        self._heap = [
            (node.last_access, next(self._tiebreak), node)
            for node in self.nodes()
            if not node.children
        ]
        heapq.heapify(self._heap)

    # ------------------------------------------------------------- sequences

    def sync_sequence(self, seq_id: Hashable, tokens: Sequence[int], num_cached: int) -> int:
        """Share a sequence's full blocks through the tree and adopt what the tree has.

        Inserts the blocks the sequence has filled, replaces any of its blocks the tree
        already had (so one physical copy survives) and extends its table with further
        cached blocks that match its tokens.

        Returns:
            The sequence's new ``num_cached`` (never less than the value passed in, and
            always leaving at least one token pending).
        """
        size = self.block_size
        limit = (len(tokens) - 1) // size
        if limit == 0:
            return num_cached
        own = self.blocks.table(seq_id)[: num_cached // size]
        canonical = self.sync(tokens, own, limit)
        self.blocks.adopt(seq_id, canonical)
        return max(num_cached, len(canonical) * size)

    def tag(self, tokens: Sequence[int], owner: SessionState) -> None:
        """Mark the cached path of ``tokens`` as belonging to ``owner``'s session.

        Called when a request of the session completes. Every node on the path is marked,
        so whichever of them is the leaf when memory runs short is scored by this session;
        a node shared with other sessions carries the mark of the latest one, and is only
        ever scored once all of their private tails are gone.
        """
        size = self.block_size
        limit = len(tokens) // size
        node, index = self.root, 0
        while index < limit:
            child = node.children.get(tuple(tokens[index * size : (index + 1) * size]))
            if child is None or not child.blocks:
                return
            matched = self._matching_blocks(child, tokens, index, limit - index)
            if matched < len(child.blocks):
                return
            child.owner = owner
            index += matched
            node = child

    # ------------------------------------------------------------- host tier

    def _gpu_path(self, tokens: Sequence[int]) -> list[RadixNode]:
        """The GPU-resident nodes whose whole edges lie along ``tokens``, root first."""
        size = self.block_size
        limit = len(tokens) // size
        path: list[RadixNode] = []
        node, index = self.root, 0
        while index < limit:
            child = node.children.get(tuple(tokens[index * size : (index + 1) * size]))
            if child is None or not child.blocks:
                break
            matched = self._matching_blocks(child, tokens, index, limit - index)
            if matched < len(child.blocks):
                break
            path.append(child)
            index += matched
            node = child
        return path

    def offload(self, tokens: Sequence[int], owner: SessionState | None = None) -> int:
        """Copy the cached KV along ``tokens`` to the host tier, where it has no copy yet.

        Called when ``owner``'s session pauses. Only blocks added since the last offload
        are copied, so the per-turn cost is the turn's new tokens. If the host pool is
        full, room is made by dropping host-only KV that is expected to be needed later
        than this session's; what still does not fit stays without a host copy (a node's
        copies always cover a prefix of its edge).

        Returns:
            The number of blocks whose copy was started.
        """
        host = self.host
        now = self.clock()
        urgency = self.retention.score(owner, now, now)
        started = 0
        for node in self._gpu_path(tokens):
            missing = node.blocks[len(node.host) :]
            if not missing:
                continue
            if host.num_free < len(missing):
                self._make_host_room(len(missing) - host.num_free, urgency)
            fits = missing[: host.num_free]
            if not fits:
                break
            slots, _ = host.offload(fits)
            node.host.extend(slots)
            started += len(fits)
            if len(fits) < len(missing):
                break
        return started

    def forget_host(self, tokens: Sequence[int]) -> None:
        """Drop the host copies along ``tokens`` (a session that will not come back)."""
        for node in self._gpu_path(tokens):
            if len(node.children) <= 1 and node.host:
                self.host.free(node.host)
                node.host = []

    def _host_leaves(self) -> list[RadixNode]:
        return [
            node
            for node in self.nodes()
            if not node.blocks and not node.children and not node.restoring
        ]

    def _make_host_room(self, num_slots: int, urgency: float) -> int:
        """Drop host-only KV expected to be needed later than ``urgency`` seconds from now.

        The entries needed last go first. Host-only KV is the only copy of its tokens, so
        it is not given up for a backup copy of something needed later than it.
        """
        retention, size, now = self.retention, self.block_size, self.clock()
        freed = 0
        while freed < num_slots:
            scored = [
                (retention.score(node.owner, node.last_used, now), -node.last_access, node)
                for node in self._host_leaves()
            ]
            scored = [entry for entry in scored if entry[0] > urgency]
            if not scored:
                break
            scored.sort(key=lambda entry: entry[:2], reverse=True)
            for _, _, node in scored:
                key = node.tokens[:size]
                count = min(len(node.host), num_slots - freed)
                self.host.free(node.host[len(node.host) - count :])
                del node.host[len(node.host) - count :]
                node.tokens = node.tokens[: len(node.host) * size]
                freed += count
                if not node.host:
                    del node.parent.children[key]
                    node.parent = None
                    self.num_nodes -= 1
                if freed >= num_slots:
                    break
        return freed

    def _drop_subtree(self, node: RadixNode) -> None:
        """Remove ``node``'s descendants (host-only by construction) and free their slots."""
        stack = list(node.children.values())
        node.children = {}
        while stack:
            child = stack.pop()
            stack.extend(child.children.values())
            if child.host:
                self.host.free(child.host)
            child.parent = None
            child.children = {}
            self.num_nodes -= 1

    def host_continuation(
        self, tokens: Sequence[int], start: int, limit: int
    ) -> list[tuple[RadixNode, int]]:
        """Host-only KV that continues a sequence's GPU prefix.

        Args:
            tokens: The sequence's tokens.
            start: Blocks the sequence already has on the GPU (its cached prefix).
            limit: Blocks it could use at most.

        Returns:
            ``(node, blocks)`` pairs in path order: the leading ``blocks`` blocks of each
            host-only node match the tokens that follow. Empty if the GPU prefix does not
            end exactly where host-only KV begins.
        """
        size = self.block_size
        node, index = self.root, 0
        while index < start:
            child = node.children.get(tuple(tokens[index * size : (index + 1) * size]))
            if child is None or not child.blocks:
                return []
            matched = self._matching_blocks(child, tokens, index, start - index)
            if matched < len(child.blocks):
                return []
            index += matched
            node = child
        plan: list[tuple[RadixNode, int]] = []
        while index < limit:
            child = node.children.get(tuple(tokens[index * size : (index + 1) * size]))
            if child is None or child.blocks or child.restoring:
                break
            matched = self._matching_blocks(child, tokens, index, limit - index)
            plan.append((child, matched))
            index += matched
            if matched * size < len(child.tokens):
                break
            node = child
        return plan

    def begin_restore(
        self, plan: list[tuple[RadixNode, int]], gpu_blocks: list[int]
    ) -> RestoreTicket:
        """Start copying the host-only KV of ``plan`` into ``gpu_blocks``.

        ``gpu_blocks`` are freshly allocated (one reference each, which the ticket keeps).
        The nodes stay host-only until :meth:`finish_restore`.
        """
        nodes: list[tuple[RadixNode, list[int]]] = []
        slots: list[int] = []
        cursor = 0
        for node, matched in plan:
            if matched * self.block_size < len(node.tokens):
                self._split(node, matched)
            node.restoring = True
            nodes.append((node, gpu_blocks[cursor : cursor + matched]))
            slots.extend(node.host)
            cursor += matched
        transfer = self.host.restore(slots, gpu_blocks)
        return RestoreTicket(nodes, list(gpu_blocks), transfer)

    def finish_restore(self, ticket: RestoreTicket) -> None:
        """Make the nodes of a completed restore GPU-resident (they keep their host copies)."""
        self._clock += 1
        self._now = self.clock()
        # A use of the restored path is a use of everything above it.
        ancestor = ticket.nodes[0][0].parent
        while ancestor is not None and ancestor is not self.root:
            self._touch(ancestor)
            ancestor = ancestor.parent
        for node, node_blocks in ticket.nodes:
            node.restoring = False
            if node.parent is None:
                continue  # dropped while the copy was in flight
            node.blocks = list(node_blocks)
            for block in node_blocks:
                self.blocks.cache_block(block)
            self.num_blocks += len(node_blocks)
            self.stats.restored_blocks += len(node_blocks)
            self._touch(node)
        ticket.finished = True

    def release_restore(self, ticket: RestoreTicket) -> None:
        """Drop the ticket's references (the sequence it was for has adopted the blocks)."""
        self.blocks.release(ticket.blocks)
        ticket.blocks = []

    # -------------------------------------------------------------- eviction

    def evict(self, num_blocks: int) -> int:
        """Free up to ``num_blocks`` blocks held only by the cache.

        Least recently used first, or — with a retention policy — the KV expected to be
        needed last first. Returns the number of blocks freed, which is less than requested
        when everything left in the tree is in use by running sequences.
        """
        if self.retention is not None:
            return self._evict_by_score(num_blocks)
        blocks = self.blocks
        freed = 0
        in_use = []
        while freed < num_blocks and self._heap:
            entry = heapq.heappop(self._heap)
            access, _, node = entry
            if node.parent is None or node.children or access != node.last_access:
                continue  # stale: removed, no longer a leaf, or touched since
            key = node.tokens[: self.block_size]
            before = freed
            while node.blocks and freed < num_blocks and blocks.ref_count(node.blocks[-1]) == 1:
                blocks.uncache_block(node.blocks.pop())
                freed += 1
            if freed > before:
                node.tokens = node.tokens[: len(node.blocks) * self.block_size]
            if node.blocks:
                # Either enough was freed, or the rest is pinned by a running sequence.
                in_use.append(entry)
                continue
            parent = node.parent
            del parent.children[key]
            node.parent = None
            self.num_nodes -= 1
            if parent is not self.root and not parent.children:
                heapq.heappush(self._heap, (parent.last_access, next(self._tiebreak), parent))
        for entry in in_use:
            heapq.heappush(self._heap, entry)
        self.num_blocks -= freed
        self.stats.evicted_blocks += freed
        return freed

    def _evict_by_score(self, num_blocks: int) -> int:
        """Eviction in the order of :meth:`SessionRetention.score`, leaves first.

        A leaf is trimmed from its tail, exactly as under LRU. When every current leaf has
        been tried and more is needed, the nodes they exposed are scored in a further round.

        With a host tier, a trimmed block that has a host copy is *demoted*: its tokens stay
        in the tree as a host-only node. Blocks without one are forgotten, and so is
        whatever host-only KV hangs below them (it could never be reached again).
        """
        blocks, retention = self.blocks, self.retention
        now = self.clock()
        freed = 0
        while freed < num_blocks:
            leaves = [
                node
                for node in self.nodes()
                if node.blocks
                and blocks.ref_count(node.blocks[-1]) == 1
                and not any(child.blocks or child.restoring for child in node.children.values())
            ]
            if not leaves:
                break
            leaves.sort(
                key=lambda n: (retention.score(n.owner, n.last_used, now), -n.last_access),
                reverse=True,
            )
            for node in leaves:
                freed += self._trim(node, num_blocks - freed)
                if freed >= num_blocks:
                    break
        self.num_blocks -= freed
        self.stats.evicted_blocks += freed
        return freed

    def _trim(self, node: RadixNode, limit: int) -> int:
        """Free up to ``limit`` unreferenced blocks from the tail of a GPU leaf's edge."""
        blocks, size = self.blocks, self.block_size
        key = node.tokens[:size]
        freed = 0
        while node.blocks and freed < limit and blocks.ref_count(node.blocks[-1]) == 1:
            blocks.uncache_block(node.blocks.pop())
            freed += 1
        if not freed:
            return 0
        kept, backed = len(node.blocks), len(node.host)
        if backed > kept:
            # Blocks [kept, backed) live on in the host tier; anything past them is forgotten.
            self.stats.demoted_blocks += backed - kept
            if backed * size < len(node.tokens):
                self._drop_subtree(node)
                node.tokens = node.tokens[: backed * size]
            if kept:
                self._split(node, kept)  # the tail (and the children) become host-only
            return freed
        if kept * size < len(node.tokens):
            # The forgotten tail takes everything below it along.
            self._drop_subtree(node)
            node.tokens = node.tokens[: kept * size]
        if not kept:
            del node.parent.children[key]
            node.parent = None
            self.num_nodes -= 1
        return freed

    def clear(self) -> int:
        """Evict everything that is not in use; return the number of blocks freed."""
        return self.evict(self.num_blocks)

    # ------------------------------------------------------------ inspection

    def nodes(self) -> Iterator[RadixNode]:
        """Every node except the root, parents before children."""
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def block_refs(self) -> dict[int, int]:
        """References the cache holds, as ``block -> 1`` (for invariant checks)."""
        return {block: 1 for node in self.nodes() for block in node.blocks}

    @property
    def num_host_blocks(self) -> int:
        """Blocks of host-only KV in the tree (restorable, not on the GPU)."""
        size = self.block_size
        return sum(len(node.tokens) // size for node in self.nodes() if not node.blocks)

    def check_invariants(self) -> None:
        """Assert the tree's structural invariants."""
        size = self.block_size
        seen: set[int] = set()
        slots: set[int] = set()
        num_nodes = 0
        for node in self.nodes():
            num_nodes += 1
            assert node.tokens and len(node.tokens) % size == 0, "edge is not block aligned"
            edge = len(node.tokens) // size
            assert node.parent is not None
            assert node.parent.children[node.tokens[:size]] is node, "node filed under wrong key"
            assert len(node.host) <= edge, "more host slots than blocks"
            for slot in node.host:
                assert slot not in slots, f"host slot {slot} appears twice in the tree"
                slots.add(slot)
            if not node.blocks:
                assert len(node.host) == edge, "host-only node without a full host copy"
                assert not any(child.blocks for child in node.children.values()), (
                    "GPU-resident node below a host-only one"
                )
                continue
            assert len(node.blocks) == edge, "GPU-resident edge is not fully on the GPU"
            assert node.last_access <= node.parent.last_access or node.parent is self.root, (
                "a child was used more recently than its parent"
            )
            for block in node.blocks:
                assert block not in seen, f"block {block} appears twice in the tree"
                assert self.blocks.is_cached(block), f"tree block {block} is not flagged cached"
                seen.add(block)
        assert num_nodes == self.num_nodes, f"{num_nodes} nodes, counter says {self.num_nodes}"
        assert len(seen) == self.num_blocks, f"{len(seen)} blocks, counter says {self.num_blocks}"
        if self.host is not None:
            assert len(slots) == self.host.num_blocks - self.host.num_free, "host slots leaked"
