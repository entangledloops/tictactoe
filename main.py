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
    def __init__(self, screen, h, w, win_len):
        self.screen = screen
        self.h = h
        self.w = w
        self.win_len = win_len
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

    def move(self, pos, value) -> bool:
        y = pos // self.w
        x = pos % self.w

        # out of bounds?
        if y < 0 or y >= self.h or x < 0 or x >= self.w:
            return False

        # is this cell already taken?
        if self.board[y][x] in SYMBOLS:
            return False

        self.board[y][x] = value
        return True

    def is_tie(self) -> bool:
        for y in range(self.h):
            for x in range(self.w):
                if self.board[y][x] not in SYMBOLS:
                    return False
        return True

    def is_winner(self, symbol) -> bool:
        win_len = self.win_len
        win = [symbol for _ in range(win_len)]
        for y in range(1 + (self.h - win_len)):
            for x in range(self.w):
                # if there's room, check across row and diag
                if (x + win_len) <= self.w:
                    cur = [self.board[y][x + i] for i in range(win_len)]
                    if cur == win:
                        return True
                    # UL -> LR
                    cur = [self.board[y + i][x + i] for i in range(win_len)]
                    if cur == win:
                        return True
                # check col
                cur = [self.board[y + i][x] for i in range(win_len)]
                if cur == win:
                    return True
                # check diag UR -> LL
                if (x + 1) >= win_len:
                    cur = [self.board[y + i][x - i] for i in range(win_len)]
                    if cur == win:
                        return True
        return False

    def draw_board(self):
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
        for row in self.board:
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

    def my_turn(self) -> bool:
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
            except ValueError:
                prompt(self.screen, f"'{inputs}' is not a valid move.")
                self.draw_board()
                continue
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

    def play(self, socket, player_id, cur_player=0):
        self.socket = socket
        self.symbol = SYMBOLS[player_id]
        while 1:
            self.draw_board()  # render current board state
            if not self.take_turn(SYMBOLS[cur_player]):
                # if something goes wrong, just quit
                break
            if self.is_game_over():
                self.handle_game_over()

            # alternate turns
            cur_player = 1 if 0 == cur_player else 0


def server(screen):
    # Computing the true max possible board size is complicated by the variable
    # width of the move numbers given a board size. Instead, we pick reasonable
    # limits. If the terminal is tiny, this can fail (even w/in provided bounds).
    # Also, 16 * 16 = 256, the full range of values in our 1 byte move protocol.
    max_h, max_w = screen.getmaxyx()
    max_h, max_w = min(max_h, 16), min(max_w, 16)
    max_dim = max(max_h, max_w)
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
    win_len = get_int(
        screen,
        f"In-a-row to win [1, {max_dim}]:",
        digits=len(str(max_dim)),
        upper_bound=max_dim,
    )
    game = Game(screen, h, w, win_len)

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
        game = Game(screen, h, w, win_len)
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
