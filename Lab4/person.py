class Person():
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
        
    def intro(self):
        # a short introduction of the person
        print('My name is ', self.name)
        print('I am ', self.age)
        
    def birthday(self):
        # adds +1 to the age
        self.age = self.age + 1
        print('It is your birthday! Congrats, now you are', self.age)
        
class Student(Person):
    
    def __init__(self, name = 'Mr. X', age = '99', study = 'nothing'):
        super().__init__(name, age)
        self.study = study