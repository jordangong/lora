import math

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from transformers.trainer_callback import TrainerCallback

from .utils import console


class RichProgressCallback(TrainerCallback):
    """Rich-based progress display for nicer training output."""

    def __init__(self):
        self.progress = None
        self.train_task = None
        self.eval_task = None
        self.max_epochs = 1
        self.in_eval = False

    @staticmethod
    def _format_epoch(epoch) -> str:
        if epoch is None:
            return "?"
        try:
            return f"{float(epoch):.2f}"
        except (TypeError, ValueError):
            return str(epoch)

    def _print_gpu_memory(self):
        """Print GPU memory usage."""
        try:
            import torch

            if not torch.cuda.is_available():
                return

            table = Table(title="GPU Memory", show_header=True, header_style="bold cyan")
            table.add_column("GPU", style="dim")
            table.add_column("Allocated", justify="right")
            table.add_column("Reserved", justify="right")
            table.add_column("Free", justify="right", style="green")
            table.add_column("Total", justify="right")

            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                free = total - reserved
                table.add_row(
                    f"gpu_{i}",
                    f"{allocated:.2f} GB",
                    f"{reserved:.2f} GB",
                    f"{free:.2f} GB",
                    f"{total:.2f} GB",
                )
            console.print(table)
        except Exception:
            pass

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize progress bar at training start."""
        self._print_gpu_memory()
        console.print(Panel("[bold green]Training Started[/bold green]", border_style="green"))

        self.max_epochs = args.num_train_epochs
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]({task.completed}/{task.total})[/dim]"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        self.progress.start()
        self.train_task = self.progress.add_task(
            f"Training [dim](epochs: {self.max_epochs:.0f})[/dim]",
            total=state.max_steps,
        )

    def on_step_end(self, args, state, control, **kwargs):
        """Update progress bar on each step."""
        if self.progress and self.train_task is not None and not self.in_eval:
            try:
                current_epoch = float(state.epoch or 0.0)
            except (TypeError, ValueError):
                current_epoch = 0.0
            max_epochs = max(1, int(round(self.max_epochs)))
            epoch = min(max_epochs, max(1, int(math.ceil(current_epoch))))
            self.progress.update(
                self.train_task,
                completed=state.global_step,
                description=f"Training [dim](epoch {epoch}/{self.max_epochs:.0f})[/dim]",
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Display training metrics inline."""
        if logs is None or self.in_eval:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith(("eval_", "_"))}
        if not train_logs:
            return

        parts = []
        for key, value in train_logs.items():
            if key == "epoch":
                continue
            if isinstance(value, float):
                if key in ["learning_rate", "total_flos"]:
                    parts.append(f"[cyan]{key}[/cyan]={value:.2e}")
                else:
                    parts.append(f"[cyan]{key}[/cyan]={value:.4f}")
            else:
                parts.append(f"[cyan]{key}[/cyan]={value}")

        epoch = logs.get("epoch", state.epoch or 0)
        if parts and self.progress:
            self.progress.console.print(
                f"  [bold]Train[/bold] @ epoch {self._format_epoch(epoch)}: " + "  ".join(parts)
            )

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        """Update eval progress bar during evaluation."""
        if self.eval_task is not None and self.progress:
            self.progress.advance(self.eval_task)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Display evaluation results."""
        self.in_eval = False

        if self.eval_task is not None and self.progress:
            self.progress.remove_task(self.eval_task)
            self.eval_task = None

        if metrics is None:
            return

        key_metrics = ["eval_loss", "eval_accuracy", "eval_perplexity"]
        parts = []
        for key in key_metrics:
            if key in metrics:
                value = metrics[key]
                name = key.replace("eval_", "")
                if isinstance(value, float):
                    parts.append(f"[green]{name}[/green]={value:.4f}")

        epoch = metrics.get("epoch", state.epoch if state is not None else "?")
        if self.progress:
            self.progress.console.print(
                f"  [bold]Eval[/bold] @ epoch {self._format_epoch(epoch)}: " + "  ".join(parts)
            )

    def on_evaluate_begin(self, args, state, control, **kwargs):
        """Add eval progress bar when evaluation starts."""
        pass

    def _start_eval_progress(self, num_steps: int):
        """Start eval progress bar with known steps."""
        if self.progress and self.eval_task is None:
            self.in_eval = True
            self.eval_task = self.progress.add_task(
                "[yellow]Evaluating[/yellow]",
                total=num_steps,
            )

    def cleanup(self):
        progress = self.progress
        self.progress = None
        self.train_task = None
        self.eval_task = None
        self.in_eval = False

        try:
            if progress is not None:
                progress.stop()
        finally:
            try:
                if progress is not None and getattr(progress, "console", None) is not None:
                    progress.console.show_cursor(True)
                else:
                    console.show_cursor(True)
            except Exception:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up progress bar and show final stats."""
        try:
            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right", style="cyan")

            if state.log_history:
                train_runtime = None
                total_flos = None
                train_loss = None
                train_samples_per_second = None

                for log in reversed(state.log_history):
                    if "train_runtime" in log:
                        train_runtime = log["train_runtime"]
                    if "total_flos" in log:
                        total_flos = log["total_flos"]
                    if "train_loss" in log:
                        train_loss = log["train_loss"]
                    if "train_samples_per_second" in log:
                        train_samples_per_second = log["train_samples_per_second"]
                    if all([train_runtime, total_flos]):
                        break

                if train_runtime:
                    hours, remainder = divmod(int(train_runtime), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    table.add_row("Training time", f"{hours:02d}:{minutes:02d}:{seconds:02d}")

                if train_loss is not None:
                    table.add_row("Final loss", f"{train_loss:.4f}")

                if train_samples_per_second:
                    table.add_row("Samples/second", f"{train_samples_per_second:.2f}")

                if total_flos:
                    table.add_row("Total FLOPs", f"{total_flos:.2e}")

            table.add_row("Total steps", str(state.global_step))
            table.add_row("Epochs completed", self._format_epoch(state.epoch))

            console.print(
                Panel(
                    table,
                    title="[bold green]✓ Training Complete[/bold green]",
                    border_style="green",
                )
            )
        finally:
            self.cleanup()
