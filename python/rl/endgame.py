from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .game import GameState, Move, Stone


@dataclass(frozen=True)
class BlockInfo:
    color: Stone
    stones: tuple[int, ...]
    liberties: frozenset[int]
    liberty_regions: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RegionInfo:
    points: tuple[int, ...]
    borders: frozenset[Stone]
    adjacent_blocks: frozenset[int]


@dataclass(frozen=True)
class BoardAnalysis:
    blocks: tuple[BlockInfo, ...]
    regions: tuple[RegionInfo, ...]
    adjacency: tuple[tuple[int, ...], ...]
    vertex_to_block: tuple[int, ...]
    vertex_to_region: tuple[int, ...]


@dataclass(frozen=True)
class TerminalResolution:
    board: list[Stone]
    removed_stones: int
    rule_resolved_groups: int
    local_search_resolved_groups: int
    preserved_seki_groups: int
    unresolved_dead_groups: int


def resolve_terminal_board(state: GameState) -> TerminalResolution:
    board = state.board.copy()
    removed_stones = 0
    rule_resolved_groups = 0
    local_search_resolved_groups = 0
    search_budget = [max(1, int(state.config.cleanup_local_search_nodes))]

    while True:
        analysis = analyze_board(state, board)
        alive_blocks = _alive_blocks(analysis)
        protected_blocks = set(alive_blocks)

        static_dead = [
            block_id
            for block_id, block in enumerate(analysis.blocks)
            if block_id not in protected_blocks and len(block.liberties) == 0
        ]
        if static_dead:
            removed_stones += _remove_blocks(board, analysis, static_dead)
            rule_resolved_groups += len(static_dead)
            continue

        candidates = [
            block_id
            for block_id in _candidate_block_ids(analysis, state.config.cleanup_candidate_max_liberties)
            if block_id not in protected_blocks
        ]
        if not candidates:
            break

        resolved_dead: list[int] = []
        for block_id in _ordered_candidate_block_ids(analysis, candidates):
            if search_budget[0] <= 0:
                break
            if _prove_block_dead(state, board, analysis, block_id, search_budget):
                resolved_dead.append(block_id)
        if not resolved_dead:
            break
        removed_stones += _remove_blocks(board, analysis, resolved_dead)
        local_search_resolved_groups += len(resolved_dead)

    final_analysis = analyze_board(state, board)
    final_alive_blocks = _alive_blocks(final_analysis)
    mixed_blocks = _mixed_border_blocks(final_analysis) if state.config.cleanup_preserve_seki else set()
    unresolved_dead_groups = 0
    if state.config.cleanup_mark_unresolved:
        unresolved_dead_groups = sum(
            1
            for block_id in _candidate_block_ids(final_analysis, state.config.cleanup_candidate_max_liberties)
            if block_id not in final_alive_blocks
            and block_id not in mixed_blocks
        )

    return TerminalResolution(
        board=board,
        removed_stones=removed_stones,
        rule_resolved_groups=rule_resolved_groups,
        local_search_resolved_groups=local_search_resolved_groups,
        preserved_seki_groups=len(mixed_blocks - final_alive_blocks),
        unresolved_dead_groups=unresolved_dead_groups,
    )


def analyze_board(state: GameState, board: list[Stone]) -> BoardAnalysis:
    topology = state.topology
    vertex_count = topology.vertex_count
    visited = [False] * vertex_count
    blocks_raw: list[tuple[Stone, tuple[int, ...], frozenset[int]]] = []
    vertex_to_block = [-1] * vertex_count

    for vertex in range(vertex_count):
        color = board[vertex]
        if color == Stone.EMPTY or visited[vertex]:
            continue
        queue: deque[int] = deque([vertex])
        visited[vertex] = True
        stones: list[int] = []
        liberties: set[int] = set()
        while queue:
            current = queue.popleft()
            stones.append(current)
            for neighbor in topology.adjacency[current]:
                stone = board[neighbor]
                if stone == Stone.EMPTY:
                    liberties.add(neighbor)
                elif stone == color and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        block_id = len(blocks_raw)
        ordered_stones = tuple(sorted(stones))
        for stone in ordered_stones:
            vertex_to_block[stone] = block_id
        blocks_raw.append((color, ordered_stones, frozenset(liberties)))

    visited = [False] * vertex_count
    regions: list[RegionInfo] = []
    vertex_to_region = [-1] * vertex_count
    for vertex in range(vertex_count):
        if board[vertex] != Stone.EMPTY or visited[vertex]:
            continue
        queue = deque([vertex])
        visited[vertex] = True
        points: list[int] = []
        borders: set[Stone] = set()
        adjacent_blocks: set[int] = set()
        while queue:
            current = queue.popleft()
            points.append(current)
            for neighbor in topology.adjacency[current]:
                stone = board[neighbor]
                if stone == Stone.EMPTY:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                    continue
                borders.add(stone)
                block_id = vertex_to_block[neighbor]
                if block_id >= 0:
                    adjacent_blocks.add(block_id)
        region_id = len(regions)
        ordered_points = tuple(sorted(points))
        for point in ordered_points:
            vertex_to_region[point] = region_id
        regions.append(
            RegionInfo(
                points=ordered_points,
                borders=frozenset(borders),
                adjacent_blocks=frozenset(adjacent_blocks),
            )
        )

    blocks: list[BlockInfo] = []
    for color, stones, liberties in blocks_raw:
        liberty_regions = frozenset(
            vertex_to_region[liberty] for liberty in liberties if 0 <= vertex_to_region[liberty] < len(regions)
        )
        blocks.append(BlockInfo(color=color, stones=stones, liberties=liberties, liberty_regions=liberty_regions))

    return BoardAnalysis(
        blocks=tuple(blocks),
        regions=tuple(regions),
        adjacency=tuple(tuple(neighbors) for neighbors in topology.adjacency),
        vertex_to_block=tuple(vertex_to_block),
        vertex_to_region=tuple(vertex_to_region),
    )


def _alive_blocks(analysis: BoardAnalysis) -> set[int]:
    alive = _benson_alive_blocks(analysis, Stone.BLACK)
    alive.update(_benson_alive_blocks(analysis, Stone.WHITE))
    return alive


def _benson_alive_blocks(analysis: BoardAnalysis, color: Stone) -> set[int]:
    candidate_blocks = {block_id for block_id, block in enumerate(analysis.blocks) if block.color == color}
    candidate_regions = {region_id for region_id, region in enumerate(analysis.regions) if region.borders == frozenset({color})}
    vital_regions: dict[int, set[int]] = {block_id: set() for block_id in candidate_blocks}

    for block_id in candidate_blocks:
        block = analysis.blocks[block_id]
        block_stones = set(block.stones)
        for region_id in candidate_regions:
            region = analysis.regions[region_id]
            if block_id not in region.adjacent_blocks:
                continue
            if all(any(neighbor in block_stones for neighbor in analysis.adjacency[point]) for point in region.points):
                vital_regions[block_id].add(region_id)

    changed = True
    active_blocks = set(candidate_blocks)
    active_regions = set(candidate_regions)
    while changed:
        changed = False
        for region_id in list(active_regions):
            region = analysis.regions[region_id]
            if not region.adjacent_blocks or any(block_id not in active_blocks for block_id in region.adjacent_blocks):
                active_regions.remove(region_id)
                changed = True
        for block_id in list(active_blocks):
            if len(vital_regions[block_id] & active_regions) < 2:
                active_blocks.remove(block_id)
                changed = True
    return active_blocks


def _mixed_border_blocks(analysis: BoardAnalysis) -> set[int]:
    mixed_blocks: set[int] = set()
    for region in analysis.regions:
        if len(region.borders) > 1:
            mixed_blocks.update(region.adjacent_blocks)
    return mixed_blocks


def _candidate_block_ids(analysis: BoardAnalysis, max_liberties: int) -> list[int]:
    candidates: list[int] = []
    for block_id, block in enumerate(analysis.blocks):
        if len(block.liberties) > max_liberties:
            continue
        own_regions = sum(1 for region_id in block.liberty_regions if analysis.regions[region_id].borders == frozenset({block.color}))
        mixed_regions = sum(1 for region_id in block.liberty_regions if len(analysis.regions[region_id].borders) > 1)
        if len(block.liberties) <= 1 or (own_regions == 0 and mixed_regions == 0):
            candidates.append(block_id)
    return candidates


def _ordered_candidate_block_ids(analysis: BoardAnalysis, candidates: list[int]) -> list[int]:
    def candidate_key(block_id: int) -> tuple[int, int, int]:
        block = analysis.blocks[block_id]
        own_regions = sum(1 for region_id in block.liberty_regions if analysis.regions[region_id].borders == frozenset({block.color}))
        return (len(block.liberties), own_regions, len(block.stones))

    return sorted(candidates, key=candidate_key)


def _remove_blocks(board: list[Stone], analysis: BoardAnalysis, block_ids: list[int]) -> int:
    removed = 0
    for block_id in block_ids:
        for vertex in analysis.blocks[block_id].stones:
            if board[vertex] != Stone.EMPTY:
                board[vertex] = Stone.EMPTY
                removed += 1
    return removed


def _prove_block_dead(
    state: GameState,
    board: list[Stone],
    analysis: BoardAnalysis,
    block_id: int,
    budget: list[int] | None = None,
) -> bool:
    block = analysis.blocks[block_id]
    target_color = block.color
    target_stones = block.stones
    search_state = _make_search_state(state, board, target_color)
    local_zone = _local_zone(state, analysis, block_id)
    cache: dict[tuple[tuple[int, ...], int, tuple[int, ...], int], bool] = {}
    remaining_budget = budget if budget is not None else [max(1, int(state.config.cleanup_local_search_nodes))]
    return _attacker_can_force_capture(
        state,
        search_state,
        target_stones,
        target_color,
        local_zone,
        max(1, int(state.config.cleanup_local_search_depth)),
        remaining_budget,
        cache,
    )


def _make_search_state(state: GameState, board: list[Stone], to_play: Stone) -> GameState:
    search_state = GameState(state.config)
    search_state.board = board.copy()
    search_state.to_play = to_play
    search_state.consecutive_passes = 0
    search_state.move_history = []
    search_state.board_history = [search_state.board.copy()]
    search_state.finished = False
    search_state.result = None
    search_state.hash_history = [search_state.compute_hash()]
    return search_state


def _local_zone(state: GameState, analysis: BoardAnalysis, block_id: int, radius: int = 3) -> frozenset[int]:
    topology = state.topology
    block = analysis.blocks[block_id]
    zone = set(block.stones)
    zone.update(block.liberties)
    frontier = set(block.stones)
    for _ in range(radius):
        next_frontier: set[int] = set()
        for vertex in frontier:
            for neighbor in topology.adjacency[vertex]:
                if neighbor not in zone:
                    zone.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    for region_id in block.liberty_regions:
        if 0 <= region_id < len(analysis.regions):
            region = analysis.regions[region_id]
            zone.update(region.points)
            for point in region.points:
                zone.update(topology.adjacency[point])
    return frozenset(zone)


def _attacker_can_force_capture(
    state: GameState,
    search_state: GameState,
    target_stones: tuple[int, ...],
    target_color: Stone,
    local_zone: frozenset[int],
    depth: int,
    budget: list[int],
    cache: dict[tuple[tuple[int, ...], int, tuple[int, ...], int], bool],
) -> bool:
    if _targets_captured(search_state.board, target_stones, target_color):
        return True
    if depth <= 0 or budget[0] <= 0:
        return False
    if _target_is_unconditionally_alive(state, search_state.board, target_stones, target_color):
        return False

    recent_hashes = tuple(search_state.hash_history[-2:])
    cache_key = (
        tuple(int(stone) for stone in search_state.board),
        int(search_state.to_play),
        recent_hashes,
        depth,
    )
    if cache_key in cache:
        return cache[cache_key]

    budget[0] -= 1
    attacker = Stone.opposite(target_color)
    ordered_children = _ordered_children(search_state, target_stones, target_color, local_zone, attacker)
    if not ordered_children:
        if search_state.to_play == attacker:
            cache[cache_key] = False
            return False
        result = _attacker_can_force_capture(
            state,
            _pass_search_turn(search_state),
            target_stones,
            target_color,
            local_zone,
            depth - 1,
            budget,
            cache,
        )
        cache[cache_key] = result
        return result

    if search_state.to_play == attacker:
        result = any(
            _attacker_can_force_capture(state, child, target_stones, target_color, local_zone, depth - 1, budget, cache)
            for child in ordered_children
        )
    else:
        result = all(
            _attacker_can_force_capture(state, child, target_stones, target_color, local_zone, depth - 1, budget, cache)
            for child in ordered_children
        )
    cache[cache_key] = result
    return result


def _ordered_children(
    search_state: GameState,
    target_stones: tuple[int, ...],
    target_color: Stone,
    local_zone: frozenset[int],
    attacker: Stone,
) -> list[GameState]:
    scored_children: list[tuple[float, GameState]] = []
    relevant_vertices = _relevant_search_vertices(search_state.board, search_state.topology.adjacency, target_stones, target_color)
    for vertex in (relevant_vertices & local_zone):
        if search_state.board[vertex] != Stone.EMPTY:
            continue
        move = Move.place(vertex)
        if not search_state.is_legal(move):
            continue
        child = search_state.copy()
        child.apply_move(move)
        if _targets_captured(child.board, target_stones, target_color):
            score = -10_000.0 if search_state.to_play == attacker else 10_000.0
            scored_children.append((score, child))
            continue
        liberties = _target_group_liberties(child.board, child.topology.adjacency, target_stones, target_color)
        if search_state.to_play == attacker:
            score = float(len(liberties))
        else:
            score = float(-len(liberties))
        scored_children.append((score, child))
    scored_children.sort(key=lambda item: item[0])
    return [child for _, child in scored_children]


def _pass_search_turn(search_state: GameState) -> GameState:
    passed = search_state.copy()
    passed.to_play = Stone.opposite(search_state.to_play)
    passed.consecutive_passes = 0
    passed.board_history.append(passed.board.copy())
    passed.hash_history.append(passed.compute_hash())
    return passed


def _targets_captured(board: list[Stone], target_stones: tuple[int, ...], target_color: Stone) -> bool:
    return any(board[vertex] != target_color for vertex in target_stones)


def _target_is_unconditionally_alive(
    state: GameState,
    board: list[Stone],
    target_stones: tuple[int, ...],
    target_color: Stone,
) -> bool:
    if _targets_captured(board, target_stones, target_color):
        return False
    analysis = analyze_board(state, board)
    anchor = target_stones[0]
    block_id = analysis.vertex_to_block[anchor]
    if block_id < 0:
        return False
    if any(analysis.vertex_to_block[vertex] != block_id for vertex in target_stones):
        return False
    return block_id in _benson_alive_blocks(analysis, target_color)


def _target_group_liberties(
    board: list[Stone],
    adjacency: list[list[int]],
    target_stones: tuple[int, ...],
    target_color: Stone,
) -> set[int]:
    _, liberties = _target_group_state(board, adjacency, target_stones, target_color)
    return liberties


def _target_group_state(
    board: list[Stone],
    adjacency: list[list[int]],
    target_stones: tuple[int, ...],
    target_color: Stone,
) -> tuple[set[int], set[int]]:
    if _targets_captured(board, target_stones, target_color):
        return set(), set()
    anchor = target_stones[0]
    queue: deque[int] = deque([anchor])
    visited = {anchor}
    liberties: set[int] = set()
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            stone = board[neighbor]
            if stone == Stone.EMPTY:
                liberties.add(neighbor)
            elif stone == target_color and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    if any(vertex not in visited for vertex in target_stones):
        return set(), set()
    return visited, liberties


def _relevant_search_vertices(
    board: list[Stone],
    adjacency: list[list[int]],
    target_stones: tuple[int, ...],
    target_color: Stone,
) -> set[int]:
    group_stones, liberties = _target_group_state(board, adjacency, target_stones, target_color)
    if not group_stones:
        return set()
    relevant = set(liberties)
    frontier = set(group_stones) | set(liberties)
    for _ in range(2):
        next_frontier: set[int] = set()
        for vertex in frontier:
            for neighbor in adjacency[vertex]:
                if board[neighbor] == Stone.EMPTY and neighbor not in relevant:
                    relevant.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    return relevant
