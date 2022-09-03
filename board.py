import torch


def is_winner(board, target, win_len) -> bool:
    h, w = board.shape[0], board.shape[1]
    for y in range(h):
        for x in range(w):
            # if there's room, check across row and diag
            if (x + win_len) <= w:
                cur = [target == board[y][x + i] for i in range(win_len)]
                if all(cur):
                    return True
                # UL -> LR
                if (y + win_len) <= h:
                    cur = [target == board[y + i][x + i] for i in range(win_len)]
                    if all(cur):
                        return True
            # check col
            if (y + win_len) <= h:
                cur = [target == board[y + i][x] for i in range(win_len)]
                if all(cur):
                    return True
                # UR -> LL
                if (x + 1) >= win_len:
                    cur = [target == board[y + i][x - i] for i in range(win_len)]
                    if all(cur):
                        return True
    return False


def is_tie(board) -> bool:
    return not torch.any(torch.zeros(board.shape, dtype=torch.float32) == board)


def generate_boards(board, symbol):
    """Generate all of the possible next moves on the board."""
    moves = torch.where(torch.zeros(board.shape, dtype=torch.float32) == board)
    if len(moves) != 2:
        return []

    boards = []
    ys, xs = moves
    for y, x in zip(ys, xs):
        new_board = board.clone()
        new_board[y.item()][x.item()] = symbol
        boards.append(new_board)
    return boards


def get_move(board_1, board_2):
    """Given two board, return the first position where they differ."""
    y, x = torch.where(board_1 != board_2)
    return y[0].item(), x[0].item()


def _minimax(db, win_len, board, prev_symbol, target_symbol):
    board_str = str(board)
    known_result = db.get(board_str, None)
    if known_result:
        return known_result
    if is_winner(board, prev_symbol, win_len):
        val = 1 if prev_symbol == target_symbol else -1
        db[board_str] = val
        return val
    elif is_tie(board):
        val = 0
        db[board_str] = val
        return val

    cur_symbol = 1 if prev_symbol == -1 else -1
    boards = generate_boards(board, cur_symbol)
    is_max_turn = cur_symbol == target_symbol
    if is_max_turn:
        max_val = -1
        for b in boards:
            val = _minimax(db, win_len, b, cur_symbol, target_symbol)
            if val >= max_val:
                max_val = val
        db[board_str] = max_val
        return max_val
    else:
        min_val = 1
        for b in boards:
            val = _minimax(db, win_len, b, cur_symbol, target_symbol)
            min_val = min(min_val, val)
        db[board_str] = min_val
        return min_val


def minimax(win_len, h, w):
    db = {}
    with torch.no_grad():
        start_board = torch.zeros(h, w, dtype=torch.float32)
        _minimax(db, win_len, start_board, -1, 1)
    return db


if __name__ == "__main__":
    t = torch.ones((3, 3), dtype=torch.float32)
    assert is_winner(t, 1, 3)
    assert not is_winner(t, -1, 3)
    assert is_tie(t)

    win_len, h, w = 3, 3, 3
    db = minimax(win_len, h, w)
    assert 5478 == len(db)
