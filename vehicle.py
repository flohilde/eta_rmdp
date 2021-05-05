from customer import Customer
from restaurant import Restaurant
from parameters import velocity
import numpy as np
import copy


class Vehicle:
    r"""
    Vehicle class that models a vehicle given by a name, a current location, a current destination, past destinations,
    current tentative route, time needed to perform the current action in the route, time when the current action was
    started, the load of the vehicle, the parking time distribution, the travel time distribution.
    """

    def __init__(self, name, location, parking_time_generator, travel_time_generator):
        r"""
        Initialize the vehicle.

        Params
        =======
            name (string):  Name of the vehicle.
            location (2D array): Current location of the vehicle.
            parking_time_generator (callable): Distribution of parking times.
            travel_time_generator (callable): Distribution of travel times.

        Attributes
        ===========
            destination (2D array): Location of the next destination.
            location_log (List): Saves all visited locations in order of visit.
            route (List): Tentative route given by a sequence of actions.
            action_time (float): Time required to perform the current action.
            started_action_time (float): Time when the current action was started.
            load (List): List of meals given by customer name that were picked up but not yet delivered.
            arrival_times (List): List of realized arrival times at restaurants.
            departure_times (List): List of realized departure times at restaurants.
            estim_times (List): Expected wait times at restaurants (expected departure time - expected arrival time).

        """
        self.name = name
        self.location = location
        self.destination = None
        self.location_log = [location]
        self.route = []
        self.action_time = 0
        self.started_action_time = 0
        self.load = []
        assert callable(parking_time_generator)
        self.parking_time_generator = parking_time_generator
        assert callable(travel_time_generator)
        self.travel_time_generator = travel_time_generator
        # Save interactions with restaurants. Estim_times saves the expected difference of departure and arrival time.
        self.arrival_times = []
        self.departure_times = []
        self.estim_times = []

    def full_deepcopy(self, customer_dict, restaurant_dict):
        r"""
        Copies the vehicle with all its relevant information.

        Arguments
        ==========
        customer_dict (Dict): Customer information considered when copying.
        restaurant_dict (Dict): Restaurant information considered when copying.

        Returns
        ========
        Copy of self.

        """
        result = Vehicle(name=self.name,
                         location=self.location,
                         parking_time_generator=self.parking_time_generator,
                         travel_time_generator=self.travel_time_generator)
        result.destination = self.destination
        result.location_log = copy.deepcopy(self.location_log)
        result.route = [action.full_deepcopy(customer_dict=customer_dict,
                                             restaurant_dict=restaurant_dict)
                        for action in self.route]
        result.action_time = self.action_time
        result.started_action_time = self.started_action_time
        result.load = copy.deepcopy(self.load)
        return result

    def act(self, current_time):
        r"""
        Acts according to its status::
            0 - Vehicle is idle.
            1 - Vehicle is driving to restaurant.
            2 - Vehicle is driving to customer.
            3 - Vehicle waits at restaurant.
            4 - Vehicle waits at customer.

        Arguments
        ==========
        current_time (int): Current time.

        Returns
        ========
        None.

        """
        # Case 1: Vehicle is idle
        if len(self.route) == 0:
            self.started_action_time = 0
            assert self.action_time == 0
            return None

        # Case 2: Vehicle just got its first action to perform
        if self.action_time == np.inf:
            # save when the action was started
            self.started_action_time = current_time
            # adjust action_time depending on new action
            new_action = self.route[0]
            if new_action.status in [1, 2]:
                self.destination = new_action.destination
                self.action_time = self.travel_time_generator(velocity=velocity, origin=self.location,
                                                              destination=self.destination)
            elif new_action.status == 3:
                self.action_time = max(self.parking_time_generator(),
                                       new_action.restaurant.get_waiting_time(orders=new_action.orders,
                                                                              current_time=current_time))
            elif new_action.status == 4:
                self.action_time = self.parking_time_generator()
            return None

        # Update action time
        self.action_time -= 1
        if self.action_time < 0:
            self.action_time = 0
        current_action = self.route[0]
        if current_action.status == 3:  # update waiting time until orders are finished at restaurant
            self.action_time = max(current_action.restaurant.get_waiting_time(orders=current_action.orders,
                                                                              current_time=current_time),
                                   self.action_time)

        if self.action_time == 0:  # last action has been completed
            # document time at which the new action is started
            self.started_action_time = current_time
            # update location if last action was a driving action
            last_action = self.route.pop(0)
            if last_action.status in [1, 2]:
                self.location = self.destination
                self.location_log.append(self.location)
                if last_action.status == 1:
                    self.arrival_times.append(current_time)

            # update load if the last action was a waiting action at the restaurant or customer
            elif last_action.status == 3:  # pick up at restaurant
                self.pickup_food(restaurant=last_action.restaurant, orders=last_action.orders)
                self.departure_times.append(current_time)
                self.estim_times.append(last_action.est_action_time)
            elif last_action.status == 4:  # drop off at customer
                self.deliver_food(customer=last_action.customer, current_time=current_time)

            # adjust action_time depending on new action
            if len(self.route) > 0:  # only if there is a next action
                new_action = self.route[0]
                if new_action.status in [1, 2]:
                    self.destination = new_action.destination
                    self.action_time = self.travel_time_generator(velocity=velocity, origin=self.location,
                                                                  destination=self.destination)

                elif new_action.status == 3:
                    self.action_time = max(self.parking_time_generator(),
                                           new_action.restaurant.get_waiting_time(orders=new_action.orders,
                                                                                  current_time=current_time))
                elif new_action.status == 4:
                    self.action_time = self.parking_time_generator()

    def pickup_food(self, restaurant, orders):
        r"""
        Picks up food at a restaurant. A food item is modeled as the customer that ordered it.

        Arguments
        ==========
        restaurant (Restaurant): Restaurant at which food is picked up by the vehicle.
        orders (List): List of customer names.

        Returns
        ========
        None.

        """
        assert isinstance(restaurant, Restaurant)
        for order in orders:
            order = restaurant.take_from_storage(order)
            if order is not None:
                self.load.append(order)
            else:
                raise Warning("Restaurant has no food for vehicle.")

    def deliver_food(self, customer, current_time):
        r"""
        Delivers food to a customer and updates the customer status.

        Arguments
        ==========
        customer (Customer): Customer to deliver food to.
        current_time (int): Current time.

        Returns
        ========
        None.

        """
        assert isinstance(customer, Customer)
        if customer.name in self.load:
            self.load.remove(customer.name)
            customer.update_status()
            customer.atd = current_time
        else:
            print([customer.name for customer in self.load])
            raise Warning("Vehicle has no food for customer {}".format(customer.name))

    def busy_time(self, time):
        r"""
        Returns the expected time the vehicle needs to finish its tentative route starting at the given time.

        Arguments
        ==========
        time (float): Time point.

        Returns
        ========
        Estimated time (float) the vehicle needs to finish its tentative route starting at the given time.

        """
        if len(self.route) == 0:
            return 0
        busy_time = max(0, self.route[0].est_action_time - (time - self.started_action_time))
        for action in self.route[1:]:
            busy_time += action.est_action_time
        if busy_time == np.inf:
            busy_time = 0
        return busy_time

