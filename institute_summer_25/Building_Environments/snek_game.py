from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.reactive import reactive
from textual.containers import Vertical
from textual import events
from rich.text import Text
from rich.style import Style
import random

class SnakeGame(Widget):
    can_focus = True
    width = 50
    height = 50

    snake = reactive([(5, 5), (5, 4), (5, 3)])
    direction = reactive("RIGHT")
    food = reactive((10, 10))
    game_over = reactive(False)

    def on_mount(self) -> None:
        self.styles.border = ("heavy", "white")
        self.box_sizing = "content-box"
        self.styles.width = self.width + 2
        self.styles.height = self.height + 2  # Border only around grid
        self.focus()
        self.set_interval(0.15, self.move_snake)

    def render(self) -> Text:
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        fx, fy = self.food
        grid[fy][fx] = "●"
        for i, (x, y) in enumerate(self.snake):
            grid[y][x] = "■" if i == 0 else "□"
        text = Text()
        for row in grid:
            for cell in row:
                if cell == "■":
                    text.append(cell, style=Style(color="green", bold=True))
                elif cell == "□":
                    text.append(cell, style=Style(color="green"))
                elif cell == "●":
                    text.append(cell, style=Style(color="red", bold=True))
                else:
                    text.append(" ")
            text.append("\n")
        return text

    def move_snake(self) -> None:
        if self.game_over:
            return
        head_x, head_y = self.snake[0]
        if self.direction == "UP":
            head_y -= 1
        elif self.direction == "DOWN":
            head_y += 1
        elif self.direction == "LEFT":
            head_x -= 1
        elif self.direction == "RIGHT":
            head_x += 1
        new_head = (head_x, head_y)
        if (
            head_x < 0 or head_x >= self.width or
            head_y < 0 or head_y >= self.height or
            new_head in self.snake
        ):
            self.game_over = True
            self.refresh()
            self.parent.query_one("#gameover").refresh()
            return
        self.snake = [new_head] + self.snake[:-1]
        if new_head == self.food:
            self.snake.append(self.snake[-1])
            self.place_food()
        self.refresh()

    def place_food(self) -> None:
        empty_spaces = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.snake
        ]
        if empty_spaces:
            self.food = random.choice(empty_spaces)
        else:
            self.game_over = True

    async def on_key(self, event: events.Key) -> None:
        if self.game_over:
            if event.key.lower() == "q":
                await self.app.action_quit()
            return
        if event.key == "up" and self.direction != "DOWN":
            self.direction = "UP"
        elif event.key == "down" and self.direction != "UP":
            self.direction = "DOWN"
        elif event.key == "left" and self.direction != "RIGHT":
            self.direction = "LEFT"
        elif event.key == "right" and self.direction != "LEFT":
            self.direction = "RIGHT"

class GameOverMessage(Widget):
    def render(self) -> Text:
        parent = self.parent.query_one(SnakeGame)
        if parent.game_over:
            msg = "Game Over! Press Q to quit."
            padding = (parent.width - len(msg)) // 2
            return Text(" " * max(padding, 0) + msg, style="bold red")
        return Text("")

class SnakeApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield SnakeGame()
            yield GameOverMessage(id="gameover")

if __name__ == "__main__":
    SnakeApp().run()
