import json
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict
import math

class Planet:
    def __init__(self, name: str, position: List[float], velocity: List[float], mass: float):
        self.name = name
        self.position: NDArray[np.float64] = np.array(position, dtype=float)
        self.velocity: NDArray[np.float64] = np.array(velocity, dtype=float)
        self.mass: float = mass
        self.acceleration: NDArray[np.float64] = np.zeros(2, dtype=float)
        self.trajectory: List[NDArray[np.float64]] = [np.array(position, dtype=float)]

    def __str__(self) -> str:
        return f"Planet: {self.name}: Position: {self.position}, Velocity: {self.velocity}, Mass: {self.mass}, Trajectory: {self.trajectory}"

def load_from_json(file_name: str) -> List[Planet]:
    """
    Loads initial attribute values of each planet from the JSON file into a dictionary and saves them by name into a list of Planet objects.

    Args:
        file_name (str): Path to the input JSON file.

    Returns:
        planets (List[Planet]): List of Planet objects.
    """
    with open(file_name, "r") as file:
        data_dict: Dict[str, Dict[str, List[float] | float]] = json.load(file)

    planets: List[Planet] = []
    for name, planet_data in data_dict.items():

        planet: Planet = Planet(name, planet_data["position"], planet_data["velocity"], planet_data["mass"])
        planets.append(planet)
    return planets

def calculate_gravitational_accelerations(planets: List[Planet]) -> None:
    """
    For each planet, calculates the vector of total gravitational acceleration caused by all the other planets.

    Args:
        planets (List[Planet]): List of Planet objects
    Returns:
        None: Doesn't return a value, only assigns each planet an acceleration attribute.
    """
    planet_count: int = len(planets)
    for i in range(planet_count):
        planet_i: Planet = planets[i]
        planets[i].acceleration = np.zeros(2, dtype=float)

        for j in range(planet_count):
            if i == j:
                continue
            planet_j: Planet = planets[j]
            r_vector: NDArray[np.float64] = planet_j.position - planet_i.position
            r: float = math.sqrt(r_vector[0] * r_vector[0] + r_vector[1] * r_vector[1])
            if r < 1e-8:
                continue
            a: float = (G * planet_j.mass) / (r * r)
            planet_i.acceleration += a * (r_vector / r )

def movement_of_planets(planets: List[Planet], dt: float) -> None:
    """
    Simulates the movement of planets based on the position, acceleration and velocity attributes.

    Args:
        planets (List[Planet]): List of Planet objects
        dt (float): Time step in seconds

    Returns:
        None: Doesn't return a value, only updates the velocity and position attributes and adds the end position into the trajectory list.    
    """
    planet_count: int = len(planets)
    for i in range(planet_count):
        planets[i].velocity += planets[i].acceleration * dt
        planets[i].position += planets[i].velocity * dt
        planets[i].trajectory.append(planets[i].position.copy())

def main_simulation_loop(planets: List[Planet], dt: float, simulation_duration: float) -> None:
    """
    Simulates the movement of planets over a time period with a specific time step.
    Utilizes the functions that calculate gravitational acceleration and movement of planets.

    Args:
        planets (List[Planet]): List of Planet objects
        dt (float): Time step in seconds
        simulation_duration (float): Duration of the simulation in seconds

    Returns:
    None: Doesn't return a value, only updates the position, velocity and trajectory attributes for each planet.
    """
    current_time: float = 0.0
    while(current_time < simulation_duration):
        calculate_gravitational_accelerations(planets)
        movement_of_planets(planets, dt)
        current_time += dt

def generate_and_save_random_properties(n: int, file_path: str) -> List[Planet]:
    """
    Generates a dictionary of n planets with random positions, velocities, and masses,
    one of them with a much higher mass to make the simulation more interesting. 
    Saves them to a JSON file and calls the load_from_json function to return the data in a list of Planet objects.

    Args:
        n (int): Number of planets to generate.
        file_path (str): Path to the output JSON file.

    Returns:
        List[Planet]: List of generated Planet objects.
    """
    planets_random_dict: Dict[str, Dict[str, List[float] | float]] = {}
    for i in range(n - 1):
        name_random: str = f"Planet_{i + 1}"
        position_random: List[float] = np.random.uniform(-3e12, 3e12, size=2).tolist()
        velocity_random: List[float] = np.random.uniform(-5000, 5000, size=2).tolist() 
        mass_random: float = float(np.random.uniform(1e23, 1e24))
        planets_random_dict[name_random] = {
            "position": position_random,
            "velocity": velocity_random,
            "mass": mass_random
        }
    name_random = "high_mass_body"
    position_high_mass: List[float] = np.random.uniform(-3e12, 3e12, size=2).tolist()
    velocity_high_mass: List[float] = np.random.uniform(-5000, 5000, size=2).tolist() 
    mass_high_mass: float = 1.989e+30
    planets_random_dict[name_random] = {
        "position": position_high_mass,
        "velocity": velocity_high_mass,
        "mass": mass_high_mass
    }
    with open(file_path, 'w') as file:
        json.dump(planets_random_dict, file, indent=4)

    return load_from_json(file_path)

G: float = 6.674e-11
"""
Gravitational constant
"""
dt: float = 3600.0
"""
Time step
"""
seconds_in_a_day: float = 3600.0 * 24.0 
