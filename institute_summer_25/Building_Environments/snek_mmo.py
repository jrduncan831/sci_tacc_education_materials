import os
import json
import time
import glob
import random
import hashlib

from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.containers import Vertical
from textual.reactive import reactive
from textual import events
from rich.text import Text
from rich.style import Style
from textual.containers import Vertical, Horizontal

# CONFIGURATION
SNEK_DATA_PATH = os.environ.get("SNEK_DATA_PATH")  # <-- CHANGE THIS TO YOUR SHARED DIR

PLAYER_ID = os.environ.get("USER", f"player{random.randint(0,9999)}")
# PLAYER_ID = f"{random.randint(0,9999)}"
PLAYER_NAME = PLAYER_ID
GRID_WIDTH = 50
GRID_HEIGHT = 25
PLAYER_TIMEOUT = 10  # seconds before a player is considered disconnected

# COLORS for up to 30 players
PLAYER_COLORS = [
    "green", "cyan", "magenta", "yellow", "blue", "red", "bright_green", "bright_cyan",
    "bright_magenta", "bright_yellow", "bright_blue", "bright_red", "white", "grey54",
    "orange3", "spring_green3", "turquoise2", "purple", "deep_pink4", "gold1",
    "chartreuse3", "aquamarine1", "medium_purple", "sandy_brown", "light_salmon1",
    "salmon1", "khaki1", "light_steel_blue", "medium_orchid", "pale_green1"
]

def get_player_color(pid):
    # Assign a color based on a hash of the player id
    idx = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % len(PLAYER_COLORS)
    return PLAYER_COLORS[idx]

def get_state_path(player_id):
    return os.path.join(SNEK_DATA_PATH, f"player_{player_id}.json")

def write_state(player_id, state):
    tmp_path = get_state_path(player_id) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, get_state_path(player_id))  # Atomic write

def read_all_states(include_inactive=False):
    now = time.time()
    states = {}
    for fname in glob.glob(os.path.join(SNEK_DATA_PATH, "player_*.json")):
        try:
            with open(fname) as f:
                state = json.load(f)
            pid = os.path.basename(fname).split("_", 1)[1].rsplit(".", 1)[0]
            # Ignore players who haven't updated recently, unless include_inactive is True
            if include_inactive or now - state.get("timestamp", 0) < PLAYER_TIMEOUT:
                states[pid] = state
        except Exception:
            continue
    return states


def initialize_food_file():
    food_path = get_food_path()
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(food_path), exist_ok=True)
    try:
        # Try to create the file exclusively; fail if it exists
        with open(food_path, 'x') as f:
            # Initialize with a random food position inside the grid
            initial_food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            json.dump({"food": initial_food, "timestamp": time.time()}, f)
    except FileExistsError:
        # File already exists, no action needed
        pass

def get_food_target(num_players, ratio):
    return max(1, num_players // ratio)

def ensure_food_on_eat(all_snakes, ratio=1):
    """Spawn new food only if food count is less than target after eating."""
    food_list = read_food()
    num_players = len(all_snakes) 
    target_count = get_food_target(num_players, ratio)
    occupied = set()
    for s in all_snakes.values():
        for pos in s["snake"]:
            occupied.add(tuple(pos))
    # Only add food if less than target
    if len(food_list) < target_count:
        empty_spaces = [
            (x, y)
            for y in range(GRID_HEIGHT)
            for x in range(GRID_WIDTH)
            if (x, y) not in occupied and (x, y) not in food_list
        ]
        if empty_spaces:
            food_list.append(random.choice(empty_spaces))
            write_food(food_list)
    # Never remove food here
    return food_list

def get_food_path():
    return os.path.join(SNEK_DATA_PATH, "food.json")

def write_food(food_list):
    tmp_path = get_food_path() + f"{PLAYER_ID}" + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"food": food_list, "timestamp": time.time()}, f)
    os.replace(tmp_path, get_food_path())

def read_food():
    try:
        with open(get_food_path()) as f:
            data = json.load(f)
        return [tuple(coord) for coord in data["food"]]
    except Exception:
        return []

def ensure_food_for_players(all_snakes, ratio=1):
    """Ensure there are enough food items for the current number of players,
    but never remove any existing food."""
    food_list = read_food()
    num_players = len(all_snakes)  # +1 for self
    target_count = max(1, num_players // ratio)
    occupied = set()
    for s in all_snakes.values():
        for pos in s["snake"]:
            occupied.add(tuple(pos))
    # Only add food if less than target
    while len(food_list) < target_count:
        empty_spaces = [
            (x, y)
            for y in range(GRID_HEIGHT)
            for x in range(GRID_WIDTH)
            if (x, y) not in occupied and (x, y) not in food_list
        ]
        if not empty_spaces:
            break
        food_list.append(random.choice(empty_spaces))
    if len(food_list) > 0:
        write_food(food_list)
    return food_list

def read_state(player_id):
    try:
        with open(get_state_path(player_id)) as f:
            return json.load(f)
    except Exception:
        return {}

def get_all_occupied_positions(all_states):
    """Return a set of all positions occupied by all snakes."""
    occupied = set()
    for state in all_states.values():
        for pos in state.get("snake", []):
            occupied.add(tuple(pos))
    return occupied

def pick_spawn_location(all_states, body_length=3):
    """Find a random, unoccupied starting location and orientation in the top left quadrant."""
    occupied = get_all_occupied_positions(all_states)
    candidates = []
    # Restrict coordinates to top left quadrant
    max_x = GRID_WIDTH // 2
    max_y = GRID_HEIGHT // 2
    for _ in range(100):  # Limit tries to avoid infinite loops
        direction = random.choice(["RIGHT", "DOWN"])
        if direction == "RIGHT":
            x = random.randint(2, max_x - 1)
            y = random.randint(0, max_y - 1)
            body = [(x, y), (x-1, y), (x-2, y)]
        elif direction == "DOWN":
            x = random.randint(0, max_x - 1)
            y = random.randint(2, max_y - 1)
            body = [(x, y), (x, y-1), (x, y-2)]
        if all((bx, by) not in occupied for (bx, by) in body):
            return body, direction
    # Fallback: just default
    if direction == "RIGHT":
        return [(0 + i, 0) for i in range(body_length)], direction
    else:
        return [(0, 0 + i) for i in range(body_length)], direction   
     
class SnakeGame(Widget):
    can_focus = True

    # Local player state
    snake = reactive([(5, 5), (5, 4), (5, 3)])
    direction = reactive("RIGHT")
    game_over = reactive(False)
    score = reactive(0)

    # Multiplayer state
    all_states = reactive({})
    food = reactive((10, 10))

    def on_mount(self) -> None:
        self.styles.border = ("heavy", "white")
        self.box_sizing = "content-box"
        self.styles.width = GRID_WIDTH + 2
        self.styles.height = GRID_HEIGHT + 2
        self.focus()
        self.food = read_food()

        # --- SNAKE SPAWN LOGIC ---
        self.all_states = read_all_states()
        # Choose a random unoccupied location in the top left quadrant!
        self.snake, self.direction = pick_spawn_location(self.all_states, body_length=3)
        self.game_over = False
        self.score = 0

        self.set_interval(0.15, self.game_tick)
        self.set_interval(0.3, self.sync_tick)

    def game_tick(self):
        if self.game_over:
            return
        self.move_snake()
        self.write_own_state()

    def sync_tick(self):
        self.all_states = read_all_states()
        # Ensure food count matches player count/ratio (adds only, never removes)
        self.food = ensure_food_for_players(self.all_states, ratio=1)  # Or ratio=3, etc.
        self.refresh()
        scoreboard = self.app.query(HighScoreBoard).first()
        if scoreboard:
            scoreboard.refresh()


    def write_own_state(self):
        # Read existing state to get high score
        existing_state = read_state(PLAYER_ID)
        high_score = existing_state.get("high_score", 0)
        # Update if current score is higher
        if self.score > high_score:
            high_score = self.score

        state = {
            "snake": self.snake,
            "direction": self.direction,
            "game_over": self.game_over,
            "score": self.score,
            "high_score": high_score,   # <--- save the highest score
            "timestamp": time.time(),
            "player_id": PLAYER_ID,
            "player_name": PLAYER_NAME,
        }
        write_state(PLAYER_ID, state)

    def move_snake(self):
        head_x, head_y = self.snake[0]
        # Calculate new head position
        if self.direction == "UP":
            head_y -= 1
        elif self.direction == "DOWN":
            head_y += 1
        elif self.direction == "LEFT":
            head_x -= 1
        elif self.direction == "RIGHT":
            head_x += 1
        new_head = (head_x, head_y)
        # Check collisions with walls or self
        if (
            head_x < 0 or head_x >= GRID_WIDTH or
            head_y < 0 or head_y >= GRID_HEIGHT or
            new_head in self.snake
        ):
            self.game_over = True
            self.write_own_state()
            self.refresh()
            self.parent.query_one("#gameover").refresh()
            return
        # Check collisions with other snakes
        for pid, state in self.all_states.items():
            if pid == PLAYER_ID:
                continue
            if tuple(new_head) in [tuple(pos) for pos in state["snake"]]:
                self.game_over = True
                self.write_own_state()
                self.refresh()
                self.parent.query_one("#gameover").refresh()
                return
        # Move snake
        self.snake = [new_head] + self.snake[:-1]

        if new_head in self.food:
            self.snake.append(self.snake[-1])
            self.score += 1
            # Remove the eaten food
            food_list = [f for f in self.food if f != new_head]
            write_food(food_list)
            # Only spawn new food if below target
            ensure_food_on_eat(self.all_states, ratio=1)  # Or ratio=3, etc.

        self.refresh()

    async def on_key(self, event: events.Key) -> None:
        if self.game_over:
            if event.key.lower() == "q":
                await self.app.action_quit()
            return
        # Change direction, disallow reverse
        if event.key == "up" and self.direction != "DOWN":
            self.direction = "UP"
        elif event.key == "down" and self.direction != "UP":
            self.direction = "DOWN"
        elif event.key == "left" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif event.key == "right" and self.direction != "LEFT":
            self.direction = "RIGHT"

    def render(self) -> Text:
        # Build a grid for the play area
        grid = [[" " for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        # Draw all snakes with user name characters as body
        all_states = dict(self.all_states)
        all_states[PLAYER_ID] = {
            "snake": self.snake,
            "player_id": PLAYER_ID,
            "player_name": PLAYER_NAME,
        }
        for pid, state in all_states.items():
            snake = state.get("snake", [])
            player_name = state.get("player_name", pid)
            color = get_player_color(pid)
            name_len = len(player_name)
            for i, (x, y) in enumerate(snake):
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                    if i == 0:
                        # Head: solid block
                        grid[y][x] = ("■", color, pid)
                    elif 1 <= i <= name_len:
                        # Body: character from player name, one per segment
                        char = player_name[i - 1]
                        grid[y][x] = (char, color, pid)
                    else:
                        # Remaining body: traditional square
                        grid[y][x] = ("□", color, pid)



        # Draw food
        for fx, fy in self.food:
            if 0 <= fx < GRID_WIDTH and 0 <= fy < GRID_HEIGHT:
                grid[fy][fx] = ("●", "red", None)

        # Render grid
        text = Text()

        for row in grid:
            for cell in row:
                if isinstance(cell, tuple):
                    ch, color, pid = cell
                    style = Style(color=color)
                    if ch == "■":
                        style = Style(color=color, bold=True)
                    text.append(ch, style=style)
                else:
                    text.append(" ")
            text.append("\n")

        return text

class GameOverMessage(Widget):
    def render(self) -> Text:
        parent = self.parent.query_one(SnakeGame)
        if parent.game_over:
            msg = f"Game Over! Score: {parent.score} | Press Q to quit."
            padding = (GRID_WIDTH - len(msg)) // 2
            return Text(" " * max(padding, 0) + msg, style="bold red")
        return Text("")

class HighScoreBoard(Widget):
    def render(self) -> Text:
        # Include inactive players to showcase all-time high scores
        states = read_all_states(include_inactive=True)
        scores = []
        for pid, state in states.items():
            name = state.get("player_name", pid)
            high_score = state.get("high_score", 0)
            scores.append((high_score, name, pid))

        # Include current player's score if not present (e.g., just started)
        if PLAYER_ID not in states:
            scores.append((0, PLAYER_NAME, PLAYER_ID))
        
        # Sort scores and show top 10
        top_scores = sorted(scores, reverse=True)[:10]
        text = Text("High Scores\n\n", style="bold underline white")

        for rank, (score, name, pid) in enumerate(top_scores, start=1):
            color = get_player_color(pid)
            name_style = Style(color=color, underline=False)
            # Number and Score in cyan, Name in player color
            text.append(f"{rank:2d}. ", style=Style(color="white", underline=False))
            text.append(f"{name[:10]:10}", style=name_style)
            text.append(f" : {score}\n", style=Style(color="white", underline=False, bold=True))
        
        return text

class MultiplayerSnakeApp(App):

    CSS = f"""
Screen {{
    align: center middle;
}}

.game-column {{
    width: {GRID_WIDTH + 2};
}}

SnakeGame {{
    width: {GRID_WIDTH + 2};
    height: {GRID_HEIGHT + 2};
}}

HighScoreBoard {{
    width: 40;
    margin-left: 1;  /* <-- reduce to 0 for no space, increase for more */
}}
"""
    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical():
                yield SnakeGame()
                yield GameOverMessage(id="gameover")
            yield HighScoreBoard()

if __name__ == "__main__":
    # Ensure the scratch directory exists
    os.makedirs(SNEK_DATA_PATH, exist_ok=True)
    initialize_food_file()
    MultiplayerSnakeApp().run()
