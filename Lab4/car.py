class Car():
    
    def __init__(self, make, model, year, color):
        # attributes of car
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        
    def describe_car(self):
        # description of the car
        print('This car is a', self.color, self.make, self.model, 'that is build in', self.year)