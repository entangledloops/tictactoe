# Copyright 2022 Stephen Dunn

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import curses
import enum
import socket

from curses.textpad import Textbox, rectangle


MY_COLOR = 1
OPPONENT_COLOR = 2
PADDING = 1
PORT = 65019  # port to listen on
SYMBOLS = ["X", "O"]


@enum.unique
class Msg(enum.Enum):
    """List of all message types for socket comms"""

    NEW_GAME = enum.auto()
    MOVE = enum.auto()
    QUIT = enum.auto()


# msg type: len(msg body)
MSG_LEN = {
    Msg.NEW_GAME: 3,
    Msg.MOVE: 1,
    Msg.QUIT: 0,
}


def get_next_symbol(symbol):
    index = (SYMBOLS.index(symbol) + 1) % len(SYMBOLS)
    return SYMBOLS[index]


def get_prev_symbol(symbol):
    index = SYMBOLS.index(symbol) - 1
    return SYMBOLS[index]


def send(socket, msg, value=None):
    try:
        socket.sendall(bytes([msg.value]))
        if value:
            socket.sendall(value)
    except:
        raise RuntimeError("Remote closed connection.")


def recv(socket) -> tuple:
    try:
        b = socket.recv(1)
        if 0 == len(b):
            raise
        msg = Msg(int(b[0]))
        value = None
        msg_len = MSG_LEN[msg]
        if msg_len:
            value = socket.recv(msg_len)
        return msg, value
    except:
        raise RuntimeError("Remote closed connection.")


def prompt(screen, text, y=PADDING, x=PADDING, clear=True, wait=True, color=None):
    if clear:
        screen.clear()
    if color:
        screen.addstr(y, x, text, color)
    else:
        screen.addstr(y, x, text)
    if not clear:
        # if the entire screen wasn't cleared, we should flush to end of line so
        # there is no weird text overlap from a previous render
        screen.clrtoeol()
    if wait:
        screen.addstr(y + 1, x, "(Press any key to continue)")
        if not clear:
            screen.clrtoeol()
    screen.refresh()
    if wait:
        screen.getch()


def get_str(screen, text, y, x, w, clear=True) -> str:
    if clear:
        screen.clear()
    curses.flushinp()  # flush pending input

    # if only inputting 1 char, just immediately return it
    if 1 == w:
        screen.addstr(y, x, text)
        if not clear:
            screen.clrtoeol()
        return chr(screen.getch())

    screen.addstr(y, x, text)
    if not clear:
        screen.clrtoeol()
    editwin = curses.newwin(1, w + 1, y + 2, x + 1)
    try:
        rectangle(screen, y + 1, x, y + 3, x + w + 2)
    except curses.error:
        pass  # ignore out of bounds drawing
    screen.refresh()

    # We use a custom validate to capture Enter and bail, rather than
    # waiting for the curses default of Ctrl+g
    def validate(ch):
        return curses.ascii.BEL if ord("\n") == ch else ch

    box = Textbox(editwin)
    box.edit(validate=validate)
    inputs = box.gather()

    return inputs.strip()


def get_int(
    screen, text, y=PADDING, x=PADDING, digits=3, lower_bound=1, upper_bound=None
) -> int:
    while 1:
        inputs = get_str(screen, text, y, x, digits)
        try:
            inputs = int(inputs)
            # convert back to string to discard leading 0s
            if len(str(inputs)) > digits:
                raise ValueError()
            if (lower_bound is not None and inputs < lower_bound) or (
                upper_bound is not None and inputs > upper_bound
            ):
                raise ValueError()
            return inputs
        except ValueError:
            if lower_bound and upper_bound:
                prompt(
                    screen,
                    f"Enter a valid number in the range [{lower_bound}, {upper_bound}].",
                )
            elif lower_bound:
                prompt(screen, f"Enter a valid number >= {lower_bound}.")
            elif upper_bound:
                prompt(screen, f"Enter a valid number <= {upper_bound}.")
            else:
                prompt(screen, "Enter a valid number.")


class Game:
    def __init__(self, screen, h, w, win_len, human):
        self.screen = screen
        self.h = h
        self.w = w
        self.win_len = win_len
        self.human = human
        self.y = PADDING
        self.x = PADDING
        self.n_cells = h * w
        self.cell_width = len(str(self.n_cells))
        self.board = []
        self.socket = None
        self.symbol = None
        self.setup_board()

    def setup_board(self):
        self.board.clear()
        cur = 1
        for row in range(self.h):
            self.board.append([])
            for _ in range(self.w):
                self.board[row].append(str(cur))
                cur += 1

    def move(self, pos, symbol, board=None) -> bool:
        if not board:
            board = self.board

        y = pos // self.w
        x = pos % self.w

        # out of bounds?
        if y < 0 or y >= self.h or x < 0 or x >= self.w:
            return False

        # is this cell already taken?
        if board[y][x] in SYMBOLS:
            return False

        board[y][x] = symbol
        return True

    def is_tie(self, board=None) -> bool:
        if not board:
            board = self.board
        for y in range(self.h):
            for x in range(self.w):
                if board[y][x] not in SYMBOLS:
                    return False
        return True

    def is_winner(self, symbol, board=None) -> bool:
        if not board:
            board = self.board
        win_len = self.win_len
        win = [symbol for _ in range(win_len)]
        for y in range(self.h):
            for x in range(self.w):
                # if there's room, check across row and diag
                if (x + win_len) <= self.w:
                    cur = [board[y][x + i] for i in range(win_len)]
                    if cur == win:
                        return True
                    # UL -> LR
                    if (y + win_len) <= self.h:
                        cur = [board[y + i][x + i] for i in range(win_len)]
                        if cur == win:
                            return True
                # check col
                if (y + win_len) <= self.h:
                    cur = [board[y + i][x] for i in range(win_len)]
                    if cur == win:
                        return True
                    # UR -> LL
                    if (x + 1) >= win_len:
                        cur = [board[y + i][x - i] for i in range(win_len)]
                        if cur == win:
                            return True
        return False

    def draw_board(self, board=None):
        if not board:
            board = self.board

        self.screen.clear()

        # draw border around board
        uly = self.y
        ulx = self.x
        lry = uly + self.h + 1
        lrx = ulx + (self.cell_width + 1) * self.w
        try:
            rectangle(self.screen, uly, ulx, lry, lrx)
        except curses.error:
            pass  # ignore out of bounds rect
        # add all values to board
        y = uly + 1
        x = ulx + 1
        for row in board:
            for col in row:
                try:
                    if self.symbol == col:
                        self.screen.addstr(y, x, col, curses.color_pair(MY_COLOR))
                    elif col in SYMBOLS:
                        self.screen.addstr(y, x, col, curses.color_pair(OPPONENT_COLOR))
                    else:
                        self.screen.addstr(y, x, col)
                except curses.error:
                    raise RuntimeError("Your terminal is too small to draw the board.")
                x += 1 + self.cell_width
            y += 1
            x = ulx + 1

        self.screen.refresh()

    def count_empty_cells(self):
        empty = 0
        for y in range(self.h):
            for x in range(self.w):
                if self.board[y][x] not in SYMBOLS:
                    empty += 1
        return empty

    def generate_boards(self, symbol, board=None):
        """Generate all of the possible next moves on the board."""
        if not board:
            board = self.board
        boards = []
        for y in range(self.h):
            for x in range(self.w):
                if board[y][x] not in SYMBOLS:
                    new_board = copy.deepcopy(board)
                    new_board[y][x] = symbol
                    boards.append(new_board)
        return boards

    def get_move(self, board_1, board_2):
        """Given two board, return the first position where they differ."""
        pos = 0
        for y in range(self.h):
            for x in range(self.w):
                if board_1[y][x] != board_2[y][x]:
                    return pos
                pos += 1

    def _minimax(self, board, prev_symbol, target_symbol, depth_limit=None, depth=0):
        if self.is_winner(prev_symbol, board):
            val = 1 if prev_symbol == target_symbol else -1
            return val, None
        elif self.is_tie(board):
            return 0, None

        # for now, default depth limit behavior is assume tie, as we can't be sure
        # that this path will lead to victory or loss
        if depth_limit and depth >= depth_limit:
            return 0 if prev_symbol == target_symbol else 1, None

        cur_symbol = get_next_symbol(prev_symbol)
        boards = self.generate_boards(cur_symbol, board)
        is_max_turn = cur_symbol == target_symbol
        if is_max_turn:
            max_val, max_board = -1, None
            for b in boards:
                val, board = self._minimax(
                    b, cur_symbol, target_symbol, depth_limit, depth + 1
                )
                if val >= max_val:
                    max_val, max_board = val, b
                # best possible move found, quit early
                if 1 == max_val:
                    break
            return max_val, max_board
        else:
            min_val, min_board = 1, None
            for b in boards:
                val, board = self._minimax(
                    b, cur_symbol, target_symbol, depth_limit, depth + 1
                )
                if val <= min_val:
                    min_val, min_board = val, b
                # opponent best possible move found, quit early
                if -1 == min_val:
                    break
            return min_val, min_board

    def minimax(self):
        """
        This minimax alg. does not re-use work from prior iterations and is a 
        minimal working implementation.

        The idea for depth limit calc. below is to consider all possibilities
        once the remaining empty board size is around the size of a typical
        tic-tac-toe board. Anything above that, we depth-limit to prevent this
        naive algorithm for blowing up time + memory.
        """
        depth_limit = None if self.count_empty_cells() < 10 else 3
        prev_symbol = get_prev_symbol(self.symbol)
        _, board = self._minimax(self.board, prev_symbol, self.symbol, depth_limit)
        return self.get_move(self.board, board)

    def my_turn(self) -> bool:
        if self.human:
            while 1:
                inputs = get_str(
                    self.screen,
                    "Enter your move (or 'q' to quit):",
                    self.y + self.h + 2,
                    self.x,
                    self.cell_width,
                    clear=False,
                )
                if "q" == inputs:
                    send(self.socket, Msg.QUIT)
                    return False
                try:
                    pos = int(inputs) - 1
                    if not self.move(pos, self.symbol):
                        raise ValueError()
                    break
                except ValueError:
                    prompt(self.screen, f"'{inputs}' is not a valid move.")
                    self.draw_board()
                    continue
        else:
            pos = self.minimax()
            if not self.move(pos, self.symbol):
                raise ValueError(f"AI picked an invalid move: {pos}")

        send(self.socket, Msg.MOVE, bytes([pos]))
        return True

    def opponents_turn(self, symbol) -> bool:
        prompt(
            self.screen,
            "Waiting for opponent...",
            self.y + self.h + 2,
            self.x,
            wait=False,
            clear=False,
        )
        msg, value = recv(self.socket)
        if Msg.QUIT == msg:
            return False
        elif Msg.MOVE == msg:
            pos = int(value[0])
            self.move(pos, symbol)
        return True

    def take_turn(self, symbol) -> bool:
        if self.symbol == symbol:
            return self.my_turn()
        else:
            return self.opponents_turn(symbol)

    def is_game_over(self) -> bool:
        winners = (self.is_winner(symbol) for symbol in SYMBOLS)
        return any(winners) or self.is_tie()

    def handle_game_over(self):
        if self.human:
            txt = "It's a tie."
            for symbol in SYMBOLS:
                if self.is_winner(symbol):
                    if self.symbol == symbol:
                        txt = "You WIN!"
                    else:
                        txt = "You LOSE!"
                    break
            self.draw_board()  # show the final board state
            prompt(
                self.screen,
                txt,
                self.y + self.h + 2,
                self.x,
                clear=False,
                color=curses.A_REVERSE,
            )
        self.setup_board()

    def play(self, socket, player_id):
        self.socket = socket
        self.symbol = SYMBOLS[player_id]
        cur_player = SYMBOLS[0]
        while 1:
            self.draw_board()  # render current board state
            if not self.take_turn(cur_player):
                # if something goes wrong, just quit
                break
            if self.is_game_over():
                self.handle_game_over()
            # alternate turns
            cur_player = get_next_symbol(cur_player)


def server(screen):
    # Computing the true max possible board size is complicated by the variable
    # width of the move numbers given a board size. Instead, we pick reasonable
    # limits. If the terminal is tiny, this can fail (even w/in provided bounds).
    # Also, 16 * 16 = 256, the full range of values in our 1 byte move protocol.
    max_h, max_w = screen.getmaxyx()
    max_h, max_w = min(max_h, 16), min(max_w, 16)
    w = get_int(
        screen,
        f"Enter board width [1, {max_w}]:",
        digits=len(str(max_w)),
        upper_bound=max_w,
    )
    h = get_int(
        screen,
        f"Enter board height [1, {max_h}]:",
        digits=len(str(max_h)),
        upper_bound=max_h,
    )
    max_dim = max(w, h)
    win_len = get_int(
        screen,
        f"In-a-row to win [1, {max_dim}]:",
        digits=len(str(max_dim)),
        upper_bound=max_dim,
    )
    human = None
    while human is None:
        human = get_str(
            screen, "Human player on this side? (y/n):", PADDING, PADDING, 1
        ).lower()
        if human not in ("y", "n"):
            human = None
        else:
            human = True if "y" == human else False

    game = Game(screen, h, w, win_len, human)
    host = ""
    prompt(screen, f"Listening on port {PORT}...", wait=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, PORT))
        s.listen()
        conn, _ = s.accept()
        params = bytes([h, w, win_len])
        send(conn, Msg.NEW_GAME, params)
        game.play(conn, 0)


def client(screen):
    host = get_str(screen, "Enter server address:", PADDING, PADDING, 15)
    prompt(screen, f"Connecting to {host}:{PORT}...", wait=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, PORT))
        msg, params = recv(s)
        if msg is not Msg.NEW_GAME:
            raise ValueError(f"Received unexpected message from server: {msg}")
        h = int(params[0])
        w = int(params[1])

        # this check doesn't guarantee rendering works, but it's better than nothing
        max_h, max_w = screen.getmaxyx()
        if h > max_h or w > max_w:
            raise ValueError(
                f"The server has a board size of {w} x {h}, but your terminal max is {max_w} x {max_h}."
            )

        win_len = int(params[2])
        game = Game(screen, h, w, win_len, human=True)
        game.play(s, 1)


def main(stdscr):
    curses.curs_set(0)  # hide cursor
    curses.init_pair(MY_COLOR, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(OPPONENT_COLOR, curses.COLOR_RED, curses.COLOR_BLACK)
    while 1:
        pressed = get_str(
            stdscr, "Are you hosting the game? (y/n/q)", PADDING, PADDING, 1
        )
        if "y" == pressed:
            server(stdscr)
            break
        elif "n" == pressed:
            client(stdscr)
            break
        elif "q" == pressed:
            break


if __name__ == "__main__":
    curses.wrapper(main)
