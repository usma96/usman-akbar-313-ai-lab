class TemperatureController:
    """
    Simulates a smart classroom temperature controller that updates its mode
    based on temperature and room occupancy percepts.
    """
    def __init__(self):
        self.previous_mode = None

    def choose_mode(self, temperature: float, occupied: bool):
        """
        Determines the target operating mode and whether a command should be sent.
        
        Parameters:
        - temperature (float): Current ambient temperature in Celsius.
        - occupied (bool): Room occupancy status.
        
        Returns:
        - tuple: (previous_mode [str], target_mode [str], send_command [bool])
        """
        # Occupancy has priority: empty room is always OFF
        if not occupied:
            target = "OFF"
        elif temperature < 18:
            target = "HEAT"
        elif temperature > 30:
            target = "COOL"
        else:
            target = "NORMAL"

        # Send a command only when the target mode changes from the previous mode
        send_command = (target != self.previous_mode)
        previous = self.previous_mode
        self.previous_mode = target

        return previous, target, send_command


# Test percepts
tests = [
    ("Repeated percept, first time", 22, True),
    ("Repeated percept, same again", 22, True),
    ("Lower exact boundary", 18, True),
    ("Upper exact boundary", 30, True),
    ("Mode transition to cooling", 31, True),
    ("Empty room overrides temperature", 25, False),
    ("Repeated empty-room percept", 25, False),
]

if __name__ == "__main__":
    controller = TemperatureController()
    print("--- Class Temperature Controller Test Execution ---")
    for label, temperature, occupied in tests:
        previous, target, send = controller.choose_mode(temperature, occupied)
        action = "SEND COMMAND" if send else "NO COMMAND"
        print(f"{label}: previous={previous}, percept=(temp={temperature}°C, occupied={occupied}), "
              f"target={target}, outcome={action}")
