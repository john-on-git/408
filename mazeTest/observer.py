class Observer:
    def __init__(self) -> None:
        pass
    def update(self, event) -> None:
        raise Exception("called abstract super method")
class Observable:
    def __init__(self) -> None:
        self.observers = [] 
    def getObservers(self):
        return self.observers
    def addObserver(self, observer : Observer) -> None:
        self.observers.append(observer)
    def removeObserver(self, observer : Observer) -> None:
        self.observers.remove(observer)
    def notify(self):
        raise Exception("called abstract super method")