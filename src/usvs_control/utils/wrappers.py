import datetime
from collections.abc import Iterable

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from usvs_control.visualization.colors import CmdColors


def progress_sim(
    iterable: Iterable[int],
    dt: float,
):
    """
    Progress bar for simulation iterations.

    Args:
        iterable: iterable to loop over
        dt (float): time step for each iteration

    Usage:
        for i in progress_sim(range(1000), dt=0.01):
            # simulation code here

    Yields:
        item from the iterable

    Raises:
        ValueError: if the iterable is empty

    """
    # Validate input
    if iterable is None or len(iterable) == 0:
        raise ValueError("The iterable must contain at least one element.")

    # Initialize progress bar
    with Progress(
        TextColumn(
            "[bold blue]{task.description}"
        ),  # custom label (defined in add_task)
        BarColumn(),  # progress bar
        "[progress.percentage]{task.percentage:>3.0f}%",  # completion percentage
        "Sim time: {task.fields[sim_time]:.2f}s",  # simulation time
        TimeElapsedColumn(),  # elapsed real time
    ) as progress:

        # Add task
        initial_time = datetime.datetime.now()
        task = progress.add_task(
            f"[{initial_time.strftime('%H:%M:%S')}] Simulation started.",
            total=len(iterable),
            sim_time=0.0,
        )

        # Iterate with progress update
        for item in iterable:
            yield item
            progress.update(task, advance=1, sim_time=item * dt)

    # Print final time
    final_time = datetime.datetime.now()
    print(
        f"{CmdColors.OKBLUE}[{final_time.strftime('%H:%M:%S')}] "
        f"Simulation completed.{CmdColors.ENDC}"
    )
