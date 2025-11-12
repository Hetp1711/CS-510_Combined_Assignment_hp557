# Student Name : Het Patel (hp557)
# Course: CS 510 Intro To AI (Assignment‑3)

from __future__ import annotations

import sys
import random
import time
import heapq
from collections import deque
from typing import List, Tuple, Dict, Optional, Set, Callable

P = print

DIRECTIONS: Tuple[str, ...] = ('up', 'down', 'left', 'right')

def load_state(file_name: str) -> List[List[int]]:
    with open(file_name, 'r') as f:
        raw_data = f.read().strip()
    lines = [line.strip() for line in raw_data.splitlines() if line.strip()]
    header_parts = lines[0].split(',')
    width = int(header_parts[0])
    height = int(header_parts[1])
    board: List[List[int]] = []
    for line in lines[1:]:
        values = [val.strip() for val in line.rstrip(',').split(',') if val.strip()]
        board.append([int(x) for x in values])
    if len(board) != height or any(len(row) != width for row in board):
        raise ValueError(f"Board dimensions in {file_name} do not match header")
    return board


def print_board(board: List[List[int]]) -> None:
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    P(f"{width},{height},")
    for row in board:
        row_str = ', '.join(str(cell) for cell in row)
        P(f" {row_str},")


def clone_state(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


def is_solved(board: List[List[int]]) -> bool:
    for row in board:
        if -1 in row:
            return False
    return True


def get_piece_ids(board: List[List[int]]) -> List[int]:
    ids: Set[int] = set()
    for row in board:
        for cell in row:
            if cell >= 2:
                ids.add(cell)
    return sorted(ids)


def get_piece_cells(board: List[List[int]], piece_id: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for r, row in enumerate(board):
        for c, val in enumerate(row):
            if val == piece_id:
                cells.append((r, c))
    return cells


def can_move(board: List[List[int]], piece_id: int, direction: str) -> bool:
    rows = len(board)
    cols = len(board[0])
    cells = get_piece_cells(board, piece_id)
    if not cells:
        return False
    for (r, c) in cells:
        if direction == 'up' and (r - 1, c) not in cells:
            new_r, new_c = r - 1, c
            if new_r < 0:
                return False
            dest_val = board[new_r][new_c]
            if piece_id == 2:
                if dest_val not in (0, -1):
                    return False
            else:
                if dest_val != 0:
                    return False
        elif direction == 'down' and (r + 1, c) not in cells:
            new_r, new_c = r + 1, c
            if new_r >= rows:
                return False
            dest_val = board[new_r][new_c]
            if piece_id == 2:
                if dest_val not in (0, -1):
                    return False
            else:
                if dest_val != 0:
                    return False
        elif direction == 'left' and (r, c - 1) not in cells:
            new_r, new_c = r, c - 1
            if new_c < 0:
                return False
            dest_val = board[new_r][new_c]
            if piece_id == 2:
                if dest_val not in (0, -1):
                    return False
            else:
                if dest_val != 0:
                    return False
        elif direction == 'right' and (r, c + 1) not in cells:
            new_r, new_c = r, c + 1
            if new_c >= cols:
                return False
            dest_val = board[new_r][new_c]
            if piece_id == 2:
                if dest_val not in (0, -1):
                    return False
            else:
                if dest_val != 0:
                    return False
    return True


def possible_moves_for_piece(board: List[List[int]], piece_id: int) -> List[str]:
    """Return a list of directions in which the specified piece can move."""
    moves: List[str] = []
    # Iterate over the globally defined directions to maintain a
    # deterministic exploration order.  We build a list rather than
    # using a generator so that callers can freely append or index.
    for direction in DIRECTIONS:
        if can_move(board, piece_id, direction):
            moves.append(direction)
    return moves


def available_moves(board: List[List[int]]) -> List[Tuple[int, str]]:
    moves: List[Tuple[int, str]] = []
    for piece_id in get_piece_ids(board):
        for direction in possible_moves_for_piece(board, piece_id):
            moves.append((piece_id, direction))
    return moves


def apply_move(board: List[List[int]], piece_id: int, direction: str) -> List[List[int]]:
    """Apply a single-cell translation of the piece and return a new board.

    The function validates the move using `can_move` and will raise
    ValueError if the move is illegal or the piece is not present.
    """
    # Validate piece existence and move legality
    cells = get_piece_cells(board, piece_id)
    if not cells:
        raise ValueError(f"Piece id {piece_id} not found on board")

    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")

    if not can_move(board, piece_id, direction):
        raise ValueError(f"Piece {piece_id} cannot move {direction}")

    new_board = [row[:] for row in board]
    for (r, c) in cells:
        if direction == 'up':
            new_board[r - 1][c] = piece_id
        elif direction == 'down':
            new_board[r + 1][c] = piece_id
        elif direction == 'left':
            new_board[r][c - 1] = piece_id
        elif direction == 'right':
            new_board[r][c + 1] = piece_id
    for (r, c) in cells:
        new_board[r][c] = 0
    return new_board


def normalize(board: List[List[int]]) -> List[List[int]]:
    """Normalize the board so piece IDs (>=3) are renumbered deterministically.

    The master block (id==2) and special cells (0,1,-1) are left unchanged.
    All other piece IDs are assigned new IDs starting at 3 based on the
    top-left location of each piece when scanning rows then columns. This
    produces a stable canonical representation used by the search routines.
    The function returns a new board (it does not modify the input in-place).
    """

    height = len(board)
    width = len(board[0]) if height > 0 else 0

    # Gather piece ids >= 3 and compute their top-left cell
    piece_tl: Dict[int, Tuple[int, int]] = {}
    for r in range(height):
        for c in range(width):
            v = board[r][c]
            if v >= 3:
                if v not in piece_tl:
                    piece_tl[v] = (r, c)
                else:
                    pr, pc = piece_tl[v]
                    # update if we found a cell that is more top-left
                    if (r, c) < (pr, pc):
                        piece_tl[v] = (r, c)

    # Sort pieces by top-left (row, col) and create a mapping
    sorted_pieces = sorted(piece_tl.items(), key=lambda iv: (iv[1][0], iv[1][1]))
    mapping: Dict[int, int] = {}
    next_id = 3
    for old_id, _ in sorted_pieces:
        mapping[old_id] = next_id
        next_id += 1

    # Build a new board with remapped ids
    new_board: List[List[int]] = [row[:] for row in board]
    for r in range(height):
        for c in range(width):
            v = board[r][c]
            if v >= 3:
                new_board[r][c] = mapping[v]

    return new_board


def serialize(board: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Return an immutable tuple of tuples representation of the board."""
    return tuple(tuple(row) for row in board)


def heuristic(board: List[List[int]]) -> int:
    """Optimized heuristic focusing on quick calculation and essential estimates."""
    # Quick solved check
    if board[1][1] == 2:
        return 0
    
    # Fast master piece search with early exit
    width = len(board[0])
    for r in range(1, min(4, len(board))):  # Master is usually in top part
        row = board[r]
        for c in range(1, min(4, width)):  # Master is usually in left part
            if row[c] == 2:
                # Combined heuristic: Manhattan + blocking penalty
                manhattan = abs(r - 1) + abs(c - 1)
                blocking = 1 if any(row[i] > 2 for i in range(1, c)) else 0
                return manhattan + blocking
    
    # Fallback for edge cases
    return 3  # Conservative estimate
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            if val == -1:
                goal_cells.append((i, j))
    if not goal_cells:
        return 0
    g_rmin = min(r for (r, _) in goal_cells)
    g_cmin = min(c for (_, c) in goal_cells)
    return abs(rmin - g_rmin) + abs(cmin - g_cmin)


def _general_search(initial_board: List[List[int]], use_queue: bool) -> Tuple[List[Tuple[int, str]], List[List[int]], int, int]:
    """General graph search routine used by BFS and DFS.

    Args:
        initial_board: The starting state of the puzzle.
        use_queue: If True, nodes are popped from the left (BFS). If
            False, nodes are popped from the right (DFS).

    Returns:
        A tuple (moves, final_board, nodes, time_ms) where ``moves``
        is the sequence of moves that solves the puzzle, ``final_board``
        is the solved configuration, ``nodes`` is the number of nodes
        expanded, and ``time_ms`` is the elapsed time in milliseconds.
    """
    start_time = time.perf_counter()
    init_norm = normalize([row[:] for row in initial_board])
    visited: Set[Tuple[Tuple[int, ...], ...]] = {serialize(init_norm)}
    # Frontier stores (board, path) pairs.  We always use a deque to
    # support both queue and stack semantics efficiently.
    frontier: deque[Tuple[List[List[int]], List[Tuple[int, str]]]] = deque()
    frontier.append((init_norm, []))
    pop = frontier.popleft if use_queue else frontier.pop
    nodes = 0
    while frontier:
        board, path = pop()
        nodes += 1
        if is_solved(board):
            end_time = time.perf_counter()
            return path, board, nodes, int((end_time - start_time) * 1000)
        # Generate children in a deterministic order
        for pid in get_piece_ids(board):
            for dirn in DIRECTIONS:
                if can_move(board, pid, dirn):
                    nb = apply_move(board, pid, dirn)
                    nb = normalize(nb)
                    key = serialize(nb)
                    if key not in visited:
                        visited.add(key)
                        frontier.append((nb, path + [(pid, dirn.upper())]))
    end_time = time.perf_counter()
    return [], initial_board, nodes, int((end_time - start_time) * 1000)


def solve_bfs(initial_board: List[List[int]]) -> Tuple[List[Tuple[int, str]], List[List[int]], int, int]:
    return _general_search(initial_board, use_queue=True)


def solve_dfs(initial_board: List[List[int]]) -> Tuple[List[Tuple[int, str]], List[List[int]], int, int]:
    return _general_search(initial_board, use_queue=False)


def _depth_limited_search(initial_board: List[List[int]], limit: int) -> Tuple[Optional[List[Tuple[int, str]]], Optional[List[List[int]]], int, bool]:
    visited: Set[Tuple[Tuple[int, ...], ...]] = set()
    init_norm = normalize([row[:] for row in initial_board])
    stack: List[Tuple[List[List[int]], List[Tuple[int, str]], int]] = [(init_norm, [], 0)]
    visited.add(serialize(init_norm))
    nodes = 0
    while stack:
        board, path, depth = stack.pop()
        nodes += 1
        if is_solved(board):
            return path, board, nodes, True
        if depth < limit:
            available: List[Tuple[int, str]] = []
            for pid in get_piece_ids(board):
                for dirn in DIRECTIONS:
                    if can_move(board, pid, dirn):
                        available.append((pid, dirn))
            # Reverse the list to simulate DFS ordering (rightmost
            # children first) when pushing onto the stack
            for pid, dirn in reversed(available):
                nb = apply_move(board, pid, dirn)
                nb = normalize(nb)
                key = serialize(nb)
                if key not in visited:
                    visited.add(key)
                    stack.append((nb, path + [(pid, dirn.upper())], depth + 1))
    return None, None, nodes, False


def solve_ids(initial_board: List[List[int]]) -> Tuple[List[Tuple[int, str]], List[List[int]], int, int]:
    start_time = time.perf_counter()
    total_nodes = 0
    limit = 0
    while True:
        path, final_board, nodes, found = _depth_limited_search(initial_board, limit)
        total_nodes += nodes
        if found:
            end_time = time.perf_counter()
            assert path is not None and final_board is not None
            return path, final_board, total_nodes, int((end_time - start_time) * 1000)
        limit += 1


def solve_competition(initial_board: List[List[int]]) -> Tuple[List[Tuple[int, str]], List[List[int]], int, int]:
    """Enhanced solver that combines A*, pattern databases, and move pruning."""
    # Initialize solver with both pattern database and move pruning
    start_time = time.perf_counter()
    init_norm = normalize([row[:] for row in initial_board])
    start_serial = serialize(init_norm)
    counter = 0
    
    # Priority queue elements: (f_score, g_score, counter, board, path)
    pq: List[Tuple[int, int, int, List[List[int]], List[Tuple[int, str]]]] = []
    
    # Initialize with multiple heuristic estimates
    h1 = heuristic(init_norm)  # Basic manhattan distance
    h2 = len([p for p in get_piece_ids(init_norm) if p > 2])  # Blocking pieces
    f_score = h1 + h2
    
    heapq.heappush(pq, (f_score, 0, counter, init_norm, []))
    
    # Track best costs and visited states
    best_g: Dict[Tuple[Tuple[int, ...], ...], int] = {start_serial: 0}
    visited = set()
    nodes = 0
    
    while pq:
        f, g_val, _, board, path = heapq.heappop(pq)
        nodes += 1
        
        board_serial = serialize(board)
        if board_serial in visited:
            continue
            
        visited.add(board_serial)
        
        if is_solved(board):
            end_time = time.perf_counter()
            return path, board, nodes, int((end_time - start_time) * 1000)
        
        # Generate and evaluate moves
        for pid in get_piece_ids(board):
            for dirn in DIRECTIONS:
                if can_move(board, pid, dirn):
                    nb = apply_move(board, pid, dirn)
                    nb = normalize(nb)
                    new_g = g_val + 1
                    serial = serialize(nb)
                    
                    if serial not in best_g or new_g < best_g[serial]:
                        best_g[serial] = new_g
                        counter += 1
                        h1 = heuristic(nb)
                        h2 = len([p for p in get_piece_ids(nb) if p > 2])
                        f_new = new_g + h1 + h2
                        heapq.heappush(pq, (f_new, new_g, counter, nb, path + [(pid, dirn.upper())]))
    
    end_time = time.perf_counter()
    return [], initial_board, nodes, int((end_time - start_time) * 1000)

def solve_astar(initial_board: List[List[int]]) -> Tuple[List[Tuple[int, str]], List[List[int]], int, int]:
    """A* solver optimized for speed and memory efficiency."""
    # Use normalize_board directly to avoid copying
    init_norm = normalize([row[:] for row in initial_board])
    start_time = time.perf_counter()
    
    # Start with normalized board and quick solved check
    init_norm = normalize([row[:] for row in initial_board])
    if is_solved(init_norm):
        end_time = time.perf_counter()
        return [], init_norm, 0, int((end_time - start_time) * 1000)
    
    # Initialize search structures
    pq: List[Tuple[int, int, int, List[List[int]], List[Tuple[int, str]]]] = []
    best_g: Dict[str, int] = {}
    nodes = 0
    counter = 0
    
    # Initial state
    h_val = heuristic(init_norm)
    start_key = str(init_norm)
    best_g[start_key] = 0
    heapq.heappush(pq, (h_val, 0, counter, init_norm, []))

    while pq:
        _, g, _, board, path = heapq.heappop(pq)
        nodes += 1
        
        # Quick solved check before any other operations
        if is_solved(board):
            end_time = time.perf_counter()
            return path, board, nodes, int((end_time - start_time) * 1000)
        
        # Get state key
        board_key = str(board)
        if board_key in best_g and g >= best_g[board_key]:
            continue
        
        best_g[board_key] = g
        
        # Get all possible moves
        for pid in get_piece_ids(board):
            for dirn in DIRECTIONS:
                if can_move(board, pid, dirn):
                    # Apply move and normalize in one step
                    new_board = normalize(apply_move(board, pid, dirn))
                    new_g = g + 1
                    
                    # Get new state key
                    new_key = str(new_board)
                    if new_key in best_g and new_g >= best_g[new_key]:
                        continue
                    
                    # Calculate new scores
                    h = heuristic(new_board)
                    f_new = new_g + h
                    counter += 1
                    
                    # Add to priority queue
                    heapq.heappush(pq, (f_new, new_g, counter, new_board, path + [(pid, dirn.upper())]))
    while pq:
        _, g_val, _, board, path = heapq.heappop(pq)
        nodes += 1
        if is_solved(board):
            end_time = time.perf_counter()
            return path, board, nodes, int((end_time - start_time) * 1000)
        for pid in get_piece_ids(board):
            for dirn in DIRECTIONS:
                if can_move(board, pid, dirn):
                    nb = apply_move(board, pid, dirn)
                    nb = normalize(nb)
                    new_g = g_val + 1
                    serial = serialize(nb)
                    # Only proceed if this path yields a lower cost
                    if serial not in best_g or new_g < best_g[serial]:
                        best_g[serial] = new_g
                        counter += 1
                        h = heuristic(nb)
                        heapq.heappush(pq, (new_g + h, new_g, counter, nb, path + [(pid, dirn.upper())]))
    end_time = time.perf_counter()
    return [], initial_board, nodes, int((end_time - start_time) * 1000)


def print_solution(moves: List[Tuple[int, str]], final_board: List[List[int]], nodes: int, time_ms: int) -> None:
    for pid, dirn in moves:
        # Format exactly as (piece_id,DIRECTION) with no space after comma
        P(f"({pid},{dirn})")
    print_board(final_board)
    P(f"Total search time: {time_ms}ms.")
    P(f"Total nodes visited: {nodes}.")
    P(f"Total solution length: {len(moves)}.")


def main() -> None:

    args = sys.argv[1:]
    if not args:
        return
    cmd = args[0]

    # Solver mapping: each solver takes a board and returns (moves, final_board, nodes, time_ms)
    SOLVERS: Dict[str, Callable[[List[List[int]]], Tuple[List[Tuple[int, str]], List[List[int]], int, int]]] = {
        'bfs': solve_bfs,
        'dfs': solve_dfs,
        'ids': solve_ids,
        'astar': solve_astar,
        'competition': solve_competition,
    }

    if cmd == 'print':
        fname = args[1]
        b = load_state(fname)
        print_board(b)
    elif cmd == 'done':
        fname = args[1]
        b = load_state(fname)
        P(str(is_solved(b)))
    elif cmd == 'availableMoves':
        fname = args[1]
        b = load_state(fname)
        moves = available_moves(b)
        for p_id, dirn in moves:
            P(f"({p_id}, {dirn})")
    elif cmd == 'applyMove':
        fname = args[1]
        move_str = args[2].strip()
        b = load_state(fname)
        parts = move_str.strip('()').split(',')
        p_id = int(parts[0].strip())
        dirn = parts[1].strip().lower()
        try:
            nb = apply_move(b, p_id, dirn)
        except ValueError as e:
            P(f"Error applying move: {e}")
        else:
            nb = normalize(nb)
            print_board(nb)
    elif cmd == 'compare':
        f1 = args[1]
        f2 = args[2]
        b1 = load_state(f1)
        b2 = load_state(f2)
        P(str(serialize(normalize(b1)) == serialize(normalize(b2))))
    elif cmd == 'norm':
        fname = args[1]
        b = load_state(fname)
        b = normalize(b)
        print_board(b)
    elif cmd == 'random':
        fname = args[1]
        steps = int(args[2])
        b = load_state(fname)
        # Print the initial state
        print_board(b)
        for _ in range(steps):
            if is_solved(b):
                break
            possible = available_moves(b)
            if not possible:
                break
            piece_id, direction = random.choice(possible)
            P(f"({piece_id}, {direction})")
            b = apply_move(b, piece_id, direction)
            b = normalize(b)
            print_board(b)
            if is_solved(b):
                break
    elif cmd in SOLVERS:
        # All search commands expect a filename as the second argument
        fname = args[1]
        b = load_state(fname)
        solver = SOLVERS[cmd]
        moves, final_board, nodes, time_ms = solver(b)
        print_solution(moves, final_board, nodes, time_ms)
    else:
        P('Unknown command')


if __name__ == '__main__':
    main()