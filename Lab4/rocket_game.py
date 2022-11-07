# Here we are creating a new module: rocket_game
# which is importing the class rocket and instantiating an object and printing its location
# Save as rocket_game.py

from rocket import Rocket, Shuttle

rocket = Rocket()
print("The rocket is at (%d, %d)." % (rocket.x, rocket.y))

shuttle = Shuttle()
print("\nThe shuttle is at (%d, %d)." % (shuttle.x, shuttle.y))
print("The shuttle has completed %d flights." % shuttle.flights_completed)