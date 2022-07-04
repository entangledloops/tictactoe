import curses
import enum
import socket

from curses.textpad import Textbox, rectangle


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
    socket.sendall(bytes([msg.value]))
    if value:
        socket.sendall(value)


def recv(socket):
    msg = Msg(int(socket.recv(1)[0]))
    value = None
    msg_len = MSG_LEN[msg]
    if msg_len:
        value = socket.recv(msg_len)
    return msg, value


def get_input(screen, prompt, y, x, w, clear=True):
    if clear:
        screen.clear()
    curses.flushinp()  # flush pending input

    # if only inputting 1 char, just immediately return it
    if 1 == w:
        screen.addstr(y, x, prompt)
        # if the entire screen wasn't cleared, we should flush to end of line so
        # there is no weird text overlap from a previous render
        if not clear:
            screen.clrtoeol()
        screen.refresh()
        return chr(screen.getch())

    def validate(ch):
        if ord("\n") == ch:
            return curses.ascii.BEL
        return ch

    screen.addstr(y, x, prompt)
    if not clear:
        screen.clrtoeol()
    editwin = curses.newwin(1, 1 + w, y + 2, x + 1)
    try:
        rectangle(screen, y + 1, x, y + 2 + 1, x + 1 + w + 1)
    except curses.error:
        pass  # ignore out of bounds drawing
    screen.refresh()

    box = Textbox(editwin)
    box.edit(validate=validate)
    inputs = box.gather()

    return inputs


def get_number(screen, text, digits=3, lower_bound=1, upper_bound=None):
    while 1:
        x = get_input(screen, text, 1, 1, digits).strip()
        try:
            if 0 == len(x) or len(x) > digits:
                raise ValueError()
            x = int(x)
            if x < lower_bound or (upper_bound and x > upper_bound):
                raise ValueError()
        except ValueError:
            if upper_bound:
                prompt(
                    screen,
                    f"You must enter a number in the range [{lower_bound}, {upper_bound}].",
                )
            else:
                prompt(screen, f"Your number must be >= {lower_bound}.")
            continue
        break
    return x


def prompt(screen, text, y=1, x=1, clear=True, wait=True, color=None):
    if clear:
        screen.clear()
    if color:
        screen.addstr(y, x, text, color)
    else:
        screen.addstr(y, x, text)
    if not clear:
        screen.clrtoeol()
    if wait:
        screen.addstr(y + 1, x, "(Press any key to continue)")
        if not clear:
            screen.clrtoeol()
    screen.refresh()
    if wait:
        screen.getch()


class Game:
    def __init__(self, screen, h=3, w=3, win_len=3):
        self.screen = screen
        self.h = h
        self.w = w
        self.win_len = win_len
        self.n_cells = h * w
        self.cell_width = len(str(self.n_cells))
        self.board = []
        for _ in range(h):
            self.board.append([None for _ in range(self.w)])
        self.socket = None
        self.symbol = None

    def setup(self, symbol):
        self.symbol = symbol
        curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        cur = 1
        for row in range(self.h):
            for col in range(self.w):
                self.board[row][col] = str(cur)
                cur += 1

    def move(self, pos, value) -> bool:
        y = (pos - 1) // self.h
        x = (pos - 1) % self.w

        # out of bounds?
        if y < 0 or y >= self.h or x < 0 or x >= self.w:
            return False

        # is this cell already taken?
        if self.board[y][x] in SYMBOLS:
            return False

        self.board[y][x] = value
        return True

    def tie(self) -> bool:
        for y in range(self.h):
            for x in range(self.w):
                if self.board[y][x] not in SYMBOLS:
                    return False
        return True

    def winner(self, symbol) -> bool:
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

    def draw(self, uly, ulx):
        # draw border around board
        lry = uly + self.h + 1
        lrx = ulx + (self.cell_width + 1) * self.w
        try:
            rectangle(self.screen, uly, ulx, lry, lrx)
        except curses.error:
            pass  # ignore out of bounds rect
        # add all values to board
        cur_y = uly + 1
        cur_x = ulx + 1
        for row in self.board:
            for col in row:
                if self.symbol == col:
                    self.screen.addstr(cur_y, cur_x, col, curses.color_pair(1))
                elif col in SYMBOLS:
                    self.screen.addstr(cur_y, cur_x, col, curses.color_pair(2))
                else:
                    self.screen.addstr(cur_y, cur_x, col)
                cur_x += 1 + self.cell_width
            cur_y += 1
            cur_x = ulx + 1

    def take_turn(self, symbol) -> bool:
        while 1:
            self.screen.clear()
            self.draw(1, 1)
            # if it's my turn...
            if symbol == self.symbol:
                inputs = get_input(
                    self.screen,
                    "Enter your move (or 'q' to quit):",
                    self.h + 3,
                    1,
                    self.cell_width,
                    clear=False,
                ).strip()
                if "q" == inputs:
                    send(self.socket, Msg.QUIT)
                    return False
                try:
                    pos = int(inputs)
                    if not self.move(pos, symbol):
                        raise ValueError()
                except ValueError:
                    prompt(self.screen, f"'{inputs}' is not a valid move.")
                    continue
                send(self.socket, Msg.MOVE, bytes([pos]))
                break
            else:
                prompt(
                    self.screen,
                    "Waiting for opponent...",
                    self.h + 3,
                    1,
                    wait=False,
                    clear=False,
                )
                msg, value = recv(self.socket)
                if Msg.QUIT == msg:
                    return False
                elif Msg.MOVE == msg:
                    pos = int(value[0])
                    self.move(pos, symbol)
                break
        self.screen.refresh()
        return True

    def play(self, socket, player_id, cur_player=0):
        self.socket = socket
        self.setup(SYMBOLS[player_id])
        while 1:
            symbol = SYMBOLS[cur_player]
            if not self.take_turn(symbol):
                break
            winner = self.winner(symbol)
            tie = self.tie()
            if winner or tie:
                self.screen.clear()
                if tie:
                    txt = "It's a tie."
                elif self.symbol == symbol:
                    txt = "You WIN!"
                else:
                    txt = "You LOSE!"
                self.draw(1, 1)
                prompt(
                    self.screen, txt, self.h + 3, 1, clear=False, color=curses.A_REVERSE
                )
                self.setup(self.symbol)

            # alternate turns
            cur_player = 1 if 0 == cur_player else 0


def server(screen):
    # Computing the true max possible board size is complicated by the variable
    # width of the move numbers. Instead, we pick reasonable upper limits, bounded
    # by the actual size of the terminal. If the terminal is tiny, this will fail.
    max_h, max_w = screen.getmaxyx()
    # leave room for text prompts and border
    max_h, max_w = min(max_h - 5, 16), min(max_w - 1, 16) 
    w = get_number(
        screen,
        f"Enter board width [1, {max_w}]:",
        digits=len(str(max_w)),
        upper_bound=max_w,
    )
    h = get_number(
        screen,
        f"Enter board height [1, {max_h}]:",
        digits=len(str(max_h)),
        upper_bound=max_h,
    )
    win_len = get_number(
        screen,
        f"In-a-row to win [2, {max(w, h)}]:",
        digits=len(str(w * h)),
        lower_bound=2,
        upper_bound=max(w, h),
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
    host = get_input(screen, "Enter server address:", 1, 1, 15).strip()
    prompt(screen, f"Connecting to {host}:{PORT}...", wait=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, PORT))
        msg, params = recv(s)
        if msg is not Msg.NEW_GAME:
            raise ValueError(f"Received unexpected message from server: {msg}.")
        h = int(params[0])
        w = int(params[1])

        max_h, max_w = screen.getmaxyx()
        max_h, max_w = max_h - 5, max_w - 1  # leave room for prompts and padding
        if h >= max_h or w >= max_w:
            raise ValueError(f"The server has a board size of {w} x {h}, but your terminal max is {max_w} x {max_h}.")

        win_len = int(params[2])
        game = Game(screen, h, w, win_len)
        game.play(s, 1)


def main(stdscr):
    curses.curs_set(0)  # hide cursor
    while 1:
        pressed = get_input(stdscr, "Are you hosting the game? (y/n/q)", 1, 1, 1)
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
