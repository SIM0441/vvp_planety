import json
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any

class Planet:
    def __init__(self, name: str, position: List[float], velocity: List[float], mass: float):
        self.name = name
        self.position: NDArray[np.float64] = np.array(position, dtype=float)
        self.velocity: NDArray[np.float64] = np.array(velocity, dtype=float)
        self.mass = mass
        self.acceleration = np.zeros(2, dtype=float)
        self.trajectory = [np.array(position, dtype=float)]


    def __str__(self) -> str:
        return f"Planet: {self.name}: Position: {self.position}, Velocity: {self.velocity}, Mass: {self.mass}"

def load_from_json(file_name: str) -> List[Planet]:
    """
    Loads initial attribute values of each planet from the JSON file and saves them into a list of Planet objects.

    Args:
        file_name (str): Path to JSON file.

    Returns:
        planets (List[Planet]): List of Planet objects.
    """

    with open(file_name, "r") as file:
        data: Dict[str, Dict[str, Any]] = json.load(file)
    
    planets: List[Planet] = []
    for name, planet_data in data.items():

        pl: Planet = Planet(name, planet_data["position"], planet_data["velocity"], planet_data["mass"])
        planets.append(pl)
    return planets


planets = load_from_json("/home/filip/VVP-Projekt/vvp_planety/data/planets.json")
for planet in planets:
    print(planet)


G: float = 6.674e-11
"""
Gravitational constant
"""

def calculate_gravitational_accelerations(planets: List[Planet], G: float) -> None:
    """
    For each planet, calculates the vector of total gravitational acceleration caused by all the other planets.

    Args:
        planets (List[Planet]): List of Planet objects
        G: Gravitational constant

    Returns:
        None: Doesn't return a value, only assigns each planet an acceleration attribute.
    """

    planet_count: int = len(planets)

    for i in range(planet_count):
        planets[i].acceleration = np.zeros(2, dtype=float)

        for j in range(planet_count):
            if i == j:
                continue
            planet_1: Planet = planets[i]
            planet_2: Planet = planets[j]
            r_vector: NDArray = planet_2.position - planet_1.position
            r: float = np.linalg.norm(r_vector)
            if r == 0:
                continue
            direction: NDArray = r_vector / r
            a: float = (G * planet_2.mass) / r**2
            a_vector: NDArray = a * direction
            planet_1.acceleration += a_vector


def movement_of_planets(planets: List[Planet], dt: float) -> None:
    """
    Simulates the movement of planets based on the position, acceleration and velocity attributes.

    Args:
        planets (List[Planet]): List of Planet objects
        dt (float): Time step

    Returns:
        None: Doesn't return a value, only updates the velocity and position attributes and adds the end position into the trajectory list .    
    """
    planet_count: int = len(planets)
    for i in range(planet_count):
        planets[i].velocity += planets[i].acceleration * dt
        planets[i].position += planets[i].velocity * dt
        planets[i].trajectory.append(planets[i].position.copy())


def main_simulation_loop(planets, dt):
    return None