from graphs.graphs import animate_full_system, animate_detailed_solar_system, plot_detailed_solar_system, plot_full_system
from physics.physics import load_from_json, main_simulation_loop, Planet
import matplotlib.animation as animation
from typing import Dict, List, Optional
import matplotlib.image as mpimg
from numpy.typing import NDArray
import numpy as np
import os

if __name__ == "__main__":
    # Setting path to the wanted json by determininng the current directory and joining it with the data subdirectory and json file_name.
    # Checking if it's planets.json for the detailed graph/animation.
    json_filename: str = "planets.json"
    current_dir: str = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    json_path: str = os.path.join(current_dir, "data", json_filename)
    is_solar_system_file: bool = "planets.json" in os.path.basename(json_path).lower()
    print(f"Loading data from: {json_path}")
    print(f"Is it planets.json file? {'Yes' if is_solar_system_file else 'No'}")
    print("-" * 40)
    # Setting simulation parameters and logging messages
    dt: float = 36000.0
    seconds_in_a_day: float = 24.0 * 3600.0
    days_in_a_year: float = 365.25
    years_to_simulate: float = 10.0
    simulation_duration: float = seconds_in_a_day * days_in_a_year * years_to_simulate
    print("---Simulation parameters---")
    print(f"Time step (dt): {dt} s ({dt / 3600:.1f} hours)")
    print(f"Total simulation duration: {simulation_duration:.2e} s ({years_to_simulate} years)")
    time_step_iterations: int = int(simulation_duration / dt)
    print(f"Number of simulation steps: {time_step_iterations}")
    print("-" * 40)
    # Potentionaly loading images for the detailed solar system graph/animation
    images: Optional[Dict[str, NDArray[np.float32]]] = None
    if is_solar_system_file:
        print("Loading planet png's...")
        images_dir: str = os.path.join(current_dir, "images")
        planet_names: List[str] = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
        images = {}
        for name in planet_names:
            path = os.path.join(images_dir, f"{name.lower()}.png")
            if os.path.exists(path):
                images[name] = mpimg.imread(path)
                print(f"{name} [OK] ")
            else:
                print(f"{name}.png not found")
        if not images:
            print("Couldn't load any png's.")
    print("-" * 40) 
    # Using the load_from_json function, load celestial bodies data from a json file
    print(f"Loading celestial bodies from: {json_path}")
    planets: List[Planet] = load_from_json(json_path)
    if not planets:
        exit("Couldn't load any celestial bodies.")
    print(f"{len(planets)} celestial bodies loaded.")
    # Simulation, Graphs and Animations
    if planets:
        print("Starting simulation...")
        main_simulation_loop(planets, dt, simulation_duration)
        print("Simulation complete")
        print("-" * 40)
        print("Creating static graphs...")
        plot_full_system(planets, images, years_to_simulate, dt)
        print("Full graph saved to 'full_graph.png'")
        if is_solar_system_file:
            plot_detailed_solar_system(planets, images, years_to_simulate, dt)
            print("Detailed graph saved to 'detailed_graph.png'")
            print("Creating detailed animation...")
            ani_detailed: Optional[animation.FuncAnimation] = animate_detailed_solar_system(planets, images, years_to_simulate, dt)
            print("Saving detailed animation...")
            project_root: str = os.path.dirname(os.path.abspath(__file__))
            animations_dir: str = os.path.join(project_root, 'animations_graphs')
            os.makedirs(animations_dir, exist_ok=True)
            ani_detailed.save(os.path.join(animations_dir, 'detailed_animation.mp4'), fps=24)
            print("Detailed animation saved to 'detailed_animation.mp4' ")
        else:
            print("Skipping solar system detail plot and animation.")
            print("-" * 40)
        print("Creating full animation...")
        ani: Optional[animation.FuncAnimation] = animate_full_system(planets, images, years_to_simulate, dt)
        print("Saving full animation...")
        project_root: str = os.path.dirname(os.path.abspath(__file__))
        animations_dir: str = os.path.join(project_root, 'animations_graphs')
        os.makedirs(animations_dir, exist_ok=True)
        ani.save(os.path.join(animations_dir, 'full_animation.mp4'), fps=24)
        print("Full animation saved to 'full_animation.mp4' ")
        print("PROGRAM COMPLETE")
    else:
        print("Cannot start simulation because planets data was not loaded successfully.")