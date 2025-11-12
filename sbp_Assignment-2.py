#Student Name : Het Patel
#Course: CS 510 Intro To AI (Assignment-2)

import sys
import random
P = print

def load_state(file_name: str):
    # Read the entire file and strip trailing whitespace
    with open(file_name, 'r') as f:
        raw_data = f.read().strip()
    # Split into non‑empty, trimmed lines
    lines = [line.strip() for line in raw_data.splitlines() if line.strip()]
    # The first line contains the width and height
    header_parts = lines[0].split(',')
    width = int(header_parts[0])
    height = int(header_parts[1])
    board = []
    # Parse each subsequent line into a list of integers
    for line in lines[1:]:
        # Remove any trailing comma and split by commas
        values = [val.strip() for val in line.rstrip(',').split(',') if val.strip()]
        board.append([int(x) for x in values])
    # Sanity check: ensure the board has the expected dimensions
    if len(board) != height or any(len(row) != width for row in board):
        raise ValueError(f"Board dimensions in {file_name} do not match header")
    return board

def print_board(board):
    """Print the board to the console in the same format it was loaded."""
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    # Print header line with width and height
    P(f"{width},{height},")
    # Print each row prefixed with a space for readability
    for row in board:
        row_str = ', '.join(str(cell) for cell in row)
        P(f" {row_str},")

def clone_state(board):
    """Return a deep copy of the board."""
    return [row[:] for row in board]

def is_solved(board):
    """Return True if the master brick covers the goal (no -1 values remain)."""
    for row in board:
        # If any cell is -1, the goal is uncovered and the puzzle is unsolved
        if -1 in row:
            return False
    return True

def get_piece_ids(board):
    """Return a sorted list of all piece IDs (>= 2) present on the board."""
    ids = set()
    for row in board:
        for cell in row:
            if cell >= 2:
                ids.add(cell)
    return sorted(ids)

def get_piece_cells(board, piece_id):
    """Return a list of (row, column) tuples where the given piece ID appears."""
    cells = []
    for row_index, row in enumerate(board):
        for col_index, value in enumerate(row):
            if value == piece_id:
                cells.append((row_index, col_index))
    return cells

def can_move(board, piece_id: int, direction: str) -> bool:
    """Return True if the given piece can move one cell in the specified direction.

    The `direction` argument must be one of 'up', 'down', 'left' or 'right'.  The
    function checks the cells at the leading edge of the piece in that
    direction and ensures that they remain on the board and do not collide
    with walls or other pieces.  The master brick (ID 2) may move onto
    the goal cell (-1), whereas other bricks may only move onto empty cells.
    """
    rows = len(board)
    cols = len(board[0])
    cells = get_piece_cells(board, piece_id)
    if not cells:
        return False
    for (r, c) in cells:
        # For each cell, check only the cells at the front in the given direction
        if direction == 'up' and (r - 1, c) not in cells:
            new_r, new_c = r - 1, c
            # Must stay on the board
            if new_r < 0:
                return False
            dest_val = board[new_r][new_c]
            # Master brick can move onto empty or goal
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

def possible_moves_for_piece(board, piece_id: int):
    """Return a list of directions that the specified piece can move."""
    directions = []
    for direction in ('up', 'down', 'left', 'right'):
        if can_move(board, piece_id, direction):
            directions.append(direction)
    return directions

def available_moves(board):
    """Return a list of all possible (piece_id, direction) moves on the board."""
    moves = []
    for piece_id in get_piece_ids(board):
        for direction in possible_moves_for_piece(board, piece_id):
            moves.append((piece_id, direction))
    return moves

def apply_move(board, piece_id: int, direction: str):
    """Return a new board with the given piece moved one step in the given direction."""
    # Get the coordinates of all cells occupied by the piece
    cells = get_piece_cells(board, piece_id)
    # Clone the board so we don't modify the original
    new_board = [row[:] for row in board]
    # Place the piece into its new positions
    for (r, c) in cells:
        if direction == 'up':
            new_board[r - 1][c] = piece_id
        elif direction == 'down':
            new_board[r + 1][c] = piece_id
        elif direction == 'left':
            new_board[r][c - 1] = piece_id
        elif direction == 'right':
            new_board[r][c + 1] = piece_id
    # Clear the old positions
    for (r, c) in cells:
        new_board[r][c] = 0
    return new_board

def compare_states(board1, board2):
    """Return True if two boards are identical, False otherwise."""
    if len(board1) != len(board2):
        return False
    for row_index in range(len(board1)):
        if board1[row_index] != board2[row_index]:
            return False
    return True

def normalize(board):
    """Return the board in normal form by renumbering the non‑master bricks.

    The algorithm scans the board row by row and ensures that bricks appear in
    ascending order of their topmost‑leftmost cell.  When a brick with a
    higher number appears before one that should come earlier, the two
    identifiers are swapped.  The master brick (2) is never renumbered.
    """
    height = len(board)
    width = len(board[0])
    next_idx = 3
    # Traverse the board row by row
    for row in range(height):
        for col in range(width):
            val = board[row][col]
            if val == next_idx:
                next_idx += 1
            elif val > next_idx:
                # We found a brick out of order; swap all occurrences of next_idx and val
                old_id = val
                new_id = next_idx
                for i in range(height):
                    for j in range(width):
                        if board[i][j] == new_id:
                            board[i][j] = old_id
                        elif board[i][j] == old_id:
                            board[i][j] = new_id
                next_idx += 1
    return board

def random_walk(board, steps: int):
    # Print the initial state
    print_board(board)
    for _ in range(steps):
        if is_solved(board):
            break
        possible = available_moves(board)
        # No moves means we are stuck
        if not possible:
            break
        # Choose a random move (piece and direction)
        piece_id, direction = random.choice(possible)
        P(f"({piece_id}, {direction})")
        # Apply the move and normalize the resulting board
        board = apply_move(board, piece_id, direction)
        board = normalize(board)
        print_board(board)
        if is_solved(board):
            break
    return board

def main():
    args=sys.argv[1:]
    if not args:
        return
    cmd=args[0]
    if cmd=='print':
        fname=args[1]
        b=load_state(fname)
        print_board(b)
    elif cmd=='done':
        fname=args[1]
        b=load_state(fname)
        P(str(is_solved(b)))
    elif cmd=='availableMoves':
        fname=args[1]
        b=load_state(fname)
        moves=available_moves(b)
        for p,dirn in moves:
            P(f"({p}, {dirn})")
    elif cmd=='applyMove':
        fname=args[1]
        m=args[2].strip()
        b=load_state(fname)
        parts=m.strip('()').split(',')
        p=int(parts[0].strip())
        dirn=parts[1].strip()
        nb=apply_move(b,p,dirn)
        print_board(nb)
    elif cmd=='compare':
        f1=args[1]
        f2=args[2]
        b1=load_state(f1)
        b2=load_state(f2)
        P(str(compare_states(b1,b2)))
    elif cmd=='norm':
        fname=args[1]
        b=load_state(fname)
        b=normalize(b)
        print_board(b)
    elif cmd=='random':
        fname=args[1]
        n=int(args[2])
        b=load_state(fname)
        random_walk(b,n)
    else:
        P('Unknown command')

if __name__=='__main__':
    main()
