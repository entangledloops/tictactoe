import pickle
import random
import time

# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

import board

board_size = 5
win_len = 4
board_size_str = f"{win_len}_{board_size}x{board_size}"
board_db_file = f"board_{board_size_str}.db"
checkpoint_file = f"checkpoint_{board_size_str}"
tensorboard_file = f"runs/tic_tac_toe_{board_size_str}"


class Model(nn.Module):
    # square boards only
    def __init__(self, board_size) -> None:
        super().__init__()
        size = board_size * board_size
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(size, size * size, dtype=torch.float32)
        self.linear2 = nn.Linear(size * size, size, dtype=torch.float32)
        self.linear3 = nn.Linear(size, 1, dtype=torch.float32)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


def train():
    writer = SummaryWriter(tensorboard_file)
    device = torch.device("cpu")
    model = Model(board_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Calculating all boards...")
    try:
        with open(board_db_file, "rb") as fp:
            db = pickle.load(fp)
    except:
        db = board.minimax(win_len, board_size, board_size)
        with open(board_db_file, "wb") as fp:
            pickle.dump(db, fp)

    print("Loading model...")
    try:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
    except:
        epoch = 0

    def save():
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "win_len": win_len,
                "board_size": board_size,
            },
            checkpoint_file,
        )

    print("Training started")
    iters = 0
    for epoch in range(epoch, epoch + 30_000):
        running_loss = 0.0
        cur_board = torch.zeros(board_size, board_size, dtype=torch.float32)
        next_boards = board.generate_boards(cur_board, 1)
        while len(next_boards) > 0:
            for b in next_boards:
                iters += 1
                expected = torch.tensor(
                    [[db[str(b)]]],
                    dtype=torch.float32,
                    device=device,
                )
                b = torch.unsqueeze(b, 0).to(device)

                optimizer.zero_grad()
                outputs = model(b)
                loss = criterion(outputs, expected)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # now make a random move, check if game over
            # then, make a random opponent move

            cur_board = random.choice(next_boards)
            if board.is_winner(cur_board, 1, board_size) or board.is_tie(cur_board):
                break
            next_boards = board.generate_boards(cur_board, -1)
            if 0 == len(next_boards):
                break
            cur_board = random.choice(next_boards)
            if board.is_winner(cur_board, -1, board_size) or board.is_tie(cur_board):
                break
            next_boards = board.generate_boards(cur_board, 1)

        if epoch % 100 == 99:
            save()
            running_loss /= 100
            writer.add_scalar("training_loss", running_loss, epoch)
            print(f"[{epoch + 1}, {iters + 1:5d}] loss: {running_loss:.3f}")
            running_loss = 0.0

    print(f"finished after {iters} iterations")
    save()


def _eval(db, model, board_size):
    cur_board = torch.zeros(board_size, board_size, dtype=torch.float32)
    print(cur_board)
    error = 0

    predicted = model(cur_board.unsqueeze(0)).item()
    actual = db[str(cur_board)]
    error += abs(predicted - actual)
    print(f"predicted: {predicted}, actual: {actual}, error: {error}")

    cur_board[0][0] = 1
    cur_board[0][1] = -1
    print(cur_board)

    predicted = model(cur_board.unsqueeze(0)).item()
    actual = db[str(cur_board)]
    error += abs(predicted - actual)
    print(f"predicted: {predicted}, actual: {actual}, error: {error}")

    cur_board[0][0] = -1
    cur_board[0][1] = 1
    cur_board[1][1] = 1
    cur_board[1][2] = 1
    cur_board[2][2] = -1
    cur_board[2][3] = -1
    print(cur_board)

    predicted = model(cur_board.unsqueeze(0)).item()
    actual = db[str(cur_board)]
    error += abs(predicted - actual)
    print(f"predicted: {predicted}, actual: {actual}, error: {error}")

    return error / 3


def eval():
    with open(board_db_file, "rb") as fp:
        db = pickle.load(fp)

    model = Model(board_size)
    error = _eval(db, model, board_size)
    print(f"RANDOM MODEL ERROR: {error}")

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    error = _eval(db, model, board_size)
    print(f"TRAINED MODEL ERROR: {error}")


if __name__ == "__main__":
    random.seed(time.time())
    train()
    eval()
