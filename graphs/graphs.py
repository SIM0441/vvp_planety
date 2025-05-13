import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from typing import List, Dict, Optional, Union
from physics.physics import Planet
from collections.abc import Sequence


# Plotting function for the full system
def plot_full_system(planets: List[Planet], images: Optional[Dict[str, NDArray]], years_to_simulate: float, dt: float) -> None:
    """
    Plots a static graph of all celestial bodies and their trajectories after a certain time.
    Args:
        planets (List[Planet]): List of Planet objects.
        images (Dict[str, np.ndarray]): Dictionary mapping planet names to image arrays (optional).
        years_to_simulate (float): Number of years to simulate.
        dt (float): Time step in seconds.
    Returns:
        None:
    """
    if not planets:
        print("Can't plot a graph, missing planets data.")
        return None
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_title(f'Full System Graph ({years_to_simulate} years, dt = {int(dt)} s)', color='white')
    ax.set_xlabel('X Position (m)', color='white')
    ax.set_ylabel('Y Position (m)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_aspect('equal', 'box')

    all_trajectory_points: np.ndarray = np.concatenate([np.array(p.trajectory) for p in planets])
    max_abs: float = np.max(np.abs(all_trajectory_points))
    buffer: float = max_abs * 0.05
    ax.set_xlim(-max_abs - buffer, max_abs + buffer)
    ax.set_ylim(-max_abs - buffer, max_abs + buffer)

    colors: tuple[tuple[float, float, float]] = plt.get_cmap('tab20').colors

    for i, planet in enumerate(planets):
        trajectory: np.ndarray = np.array(planet.trajectory)
        color: tuple[float, float, float] = colors[i % len(colors)]
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=1, color=color, linewidth=0.35, label=planet.name)
        final_position = planet.position

        if images and planet.name in images:
            image: np.ndarray = images[planet.name]
            ab: AnnotationBbox = AnnotationBbox(OffsetImage(image, zoom=0.006), tuple(final_position), frameon=False, pad=0)
            ax.add_artist(ab)
        else:
            ax.plot(final_position[0], final_position[1], 'o', color=color, markersize=4)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', facecolor='black', edgecolor='white', labelcolor='white', fontsize=7)

    plt.tight_layout(rect=(0, 0, 0.8, 1))
    project_root = os.path.dirname(os.path.abspath(__file__))
    animations_dir = os.path.join(project_root, '..', 'animations_graphs')
    os.makedirs(animations_dir, exist_ok=True)
    fig.savefig(os.path.join(animations_dir, 'full_graph.png'), dpi=500, bbox_inches='tight', facecolor='black')

# Plotting function for the solar system detail
def plot_detailed_solar_system(planets: List[Planet], images: Optional[Dict[str, NDArray]], years_to_simulate: float, dt: float) -> None:
    """
    Special function for better seeing the position and trajectories of the first 5 celestial bodies in the Solar system.
    Args:
        planets (List[Planet]): List of Planet objects.
        images (Dict[str, np.ndarray]): Dictionary mapping planet names to image arrays (optional).
        years_to_simulate (float): Number of years to simulate.
        dt (float): Time step in seconds.
    Returns:
        None:
    """
    if not planets:
        print("Can't plot a graph, missing planets data.")
        return None
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_title(f'Detailed Solar System Graph ({years_to_simulate} years, dt = {int(dt)} s)', color='white')
    ax.set_xlabel('X Position (m)', color='white')
    ax.set_ylabel('Y Position (m)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_aspect('equal', 'box')

    inner_planet_names: List[str] = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars']
    inner_planets: List[Planet] = [p for p in planets if p.name in inner_planet_names]
    all_trajectory_points: np.ndarray = np.concatenate([np.array(p.trajectory) for p in inner_planets])
    max_abs: float = np.max(np.abs(all_trajectory_points))
    buffer: float = max_abs * 0.05
    ax.set_xlim(-max_abs - buffer, max_abs + buffer)
    ax.set_ylim(-max_abs - buffer, max_abs + buffer)

    for i, planet in enumerate(inner_planets):
        trajectory: np.ndarray = np.array(planet.trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.3, color='white', linewidth=0.25) 
        final_position: np.ndarray = planet.position
        if images and planet.name in images:
            image: np.ndarray = images[planet.name]
            image_zoom: float = 0.05 if planet.name != 'Sun' else 0.09
            imagebox: OffsetImage = OffsetImage(image, zoom=image_zoom)
            ab: AnnotationBbox = AnnotationBbox(imagebox, tuple(final_position), frameon=False, pad=0)
            ax.add_artist(ab)
        else:
            ax.plot(final_position[0], final_position[1], 'o', color='white', markersize=6)
    plt.tight_layout()
    project_root = os.path.dirname(os.path.abspath(__file__))
    animations_dir = os.path.join(project_root, '..', 'animations_graphs')
    os.makedirs(animations_dir, exist_ok=True)
    fig.savefig(os.path.join(animations_dir, 'detailed_graph.png'), dpi=500, bbox_inches='tight', facecolor='black')

# Animation function for the full system
def animate_full_system(planets: List[Planet], images: Optional[Dict[str, NDArray]], years_to_simulate: float, dt: float) -> Optional[animation.FuncAnimation]:
    """
    Plots an animation of the full solar system celestial bodies and their trajectories over a given time.
    Args:
        planets (List[Planet]): List of Planet objects representing celestial bodies in the system.
        images (Dict[str, np.ndarray] | None): Dictionary mapping planet names to image arrays (optional).
        years_to_simulate (float): Total number of simulated years.
        dt (float): Time step in seconds for the simulation.
    Returns:
        Optional[animation.FuncAnimation]: The full animation object, which can be displayed or saved, or None if input is invalid.
    """
    if not planets:
        print("Missing planets, can't animate.")
        return
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_title(f'Full System Animation ({years_to_simulate} years, dt = {int(dt)} s)', color='white')
    ax.set_xlabel('X Position (m)', color='white')
    ax.set_ylabel('Y Position (m)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_aspect('equal', 'box')

    all_trajectory_points: np.ndarray = np.concatenate([np.array(p.trajectory) for p in planets])
    max_abs: float = np.max(np.abs(all_trajectory_points))
    buffer: float = max_abs * 0.05
    ax.set_xlim(-max_abs - buffer, max_abs + buffer)
    ax.set_ylim(-max_abs - buffer, max_abs + buffer)

    planet_images: List[Union[AnnotationBbox, Line2D]] = []
    trajectories: List[Line2D] = []
    colors: tuple[tuple[float, float, float]] = plt.get_cmap('tab20').colors
    for i, planet in enumerate(planets):
        color: tuple[float, float, float] = colors[i % len(colors)]
        if images and planet.name in images:
            image: np.ndarray = images[planet.name]
            imagebox: OffsetImage = OffsetImage(image, zoom=0.006)
            ab: AnnotationBbox = AnnotationBbox(imagebox, (0, 0), frameon=False)
            ax.add_artist(ab)
            planet_images.append(ab)
        else:
            marker: Line2D
            marker, = ax.plot(*planet.trajectory[0], marker='o', color=color, markersize=6)
            planet_images.append(marker)
        line: Line2D
        line, = ax.plot([], [], '-', color=color, alpha=1, linewidth=0.4)
        trajectories.append(line)

    def update(frame: int) -> Sequence[Artist]:
        for planet, image_artist, line in zip(planets, planet_images, trajectories):
            if frame < len(planet.trajectory):
                position: np.ndarray = planet.trajectory[frame]
                if isinstance(image_artist, AnnotationBbox):
                    image_artist.xybox = (float(position[0]), float(position[1]))
                else:
                    image_artist.set_data([position[0]], [position[1]])
                traj: np.ndarray = np.array(planet.trajectory[:frame+1])
                line.set_data(traj[:, 0], traj[:, 1])
        return planet_images + trajectories
    
    number_of_frames: int = max(len(p.trajectory) for p in planets)
    frame_step = 5
    frames_for_animation = range(0, number_of_frames, frame_step)
    ani: animation.FuncAnimation = animation.FuncAnimation(fig, update, frames=frames_for_animation, interval=24, blit=True)
    return ani

# Animation function for the solar system detail
def animate_detailed_solar_system(planets: List[Planet], images: Optional[Dict[str, NDArray]] | None, years_to_simulate: float, dt: float) -> Optional[animation.FuncAnimation]:
    """
    Plots an animation of the inner solar system celestial bodies and their trajectories over a given time.
    Args:
        planets (List[Planet]): List of Planet objects representing the inner solar system.
        images (Dict[str, np.ndarray] | None): Dictionary mapping planet names to image arrays (optional).
        years_to_simulate (float): Number of years to simulate.
        dt (float): Time step in seconds.
    Returns:
        Optional[animation.FuncAnimation]: The detailed animation object, which can be displayed or saved, or None if input is invalid.
    """
    if not planets:
        print("Missing planets, can't animate.")
        return
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_title(f'Detailed Solar System Animation ({years_to_simulate} years, dt = {int(dt)} s)', color='white')
    ax.set_xlabel('X Position (m)', color='white')
    ax.set_ylabel('Y Position (m)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_aspect('equal', 'box')

    inner_planet_names: List[str] = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars']
    inner_planets: List[Planet] = [p for p in planets if p.name in inner_planet_names]
    all_trajectory_points: np.ndarray = np.concatenate([np.array(p.trajectory) for p in inner_planets])
    max_abs: float = np.max(np.abs(all_trajectory_points))
    buffer: float = max_abs * 0.05
    ax.set_xlim(-max_abs - buffer, max_abs + buffer)
    ax.set_ylim(-max_abs - buffer, max_abs + buffer)

    planet_images: List[Union[AnnotationBbox, Line2D]] = []
    trajectories: List[Line2D] = []
    colors: tuple[tuple[float, float, float]] = plt.get_cmap('tab10').colors
    for i, planet in enumerate(inner_planets):
        color: tuple[float, float, float] = colors[i % len(colors)]
        if images and planet.name in images:
            img: np.ndarray = images[planet.name]
            zoom_inset: float = 0.05 if planet.name != 'Sun' else 0.09
            imagebox: OffsetImage = OffsetImage(img, zoom=zoom_inset)
            ab: AnnotationBbox = AnnotationBbox(imagebox, (0, 0), frameon=False, pad=0)
            ax.add_artist(ab)
            planet_images.append(ab)
        else:
            marker: Line2D
            marker, = ax.plot(*planet.trajectory[0], marker='o', color=color, markersize=20)
            planet_images.append(marker)
        line: Line2D
        line, = ax.plot([], [], '-', color='white', alpha=0.3, linewidth=0.25)
        trajectories.append(line)

    def update(frame: int) -> Sequence[Artist]:
        for planet, image_artist, line in zip(inner_planets, planet_images, trajectories):
            if frame < len(planet.trajectory):
                position: np.ndarray = planet.trajectory[frame]
                if isinstance(image_artist, AnnotationBbox):
                    image_artist.xybox = (float(position[0]), float(position[1]))
                else:
                    image_artist.set_data([position[0]], [position[1]])
                traj: np.ndarray = np.array(planet.trajectory[:frame+1])
                line.set_data(traj[:, 0], traj[:, 1])
        return planet_images + trajectories
    
    number_of_frames: int = max(len(p.trajectory) for p in inner_planets)
    frame_step: int = 5
    frames_for_animation = range(0, number_of_frames, frame_step)
    ani: animation.FuncAnimation = animation.FuncAnimation(fig, update, frames=frames_for_animation, interval=24, blit=True)
    return ani