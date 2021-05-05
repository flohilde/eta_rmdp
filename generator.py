import numpy as np
from customer import Customer
from restaurant import Restaurant
from vehicle import Vehicle
import parameters as param
import simplejson as json


# global customer index
customer_index = 0


def vehicle_generator(n_vehicles):
    r"""
    Initializes n vehicles with random locations and parking and travel time distribution as specified in the
    parameter file.

    Arguments
    ==========
    n_vehicles (int): Number of vehicles to generate.

    Returns
    ========
    List of vehicles (Vehicle).

    """
    vehicles = []
    for i in range(n_vehicles):
        location = np.array([np.random.uniform(param.area_size[0][0], param.area_size[0][1]),
                             np.random.uniform(param.area_size[1][0], param.area_size[1][1])])

        mu_parking, sigma_parking = param.mean_wait_parking, param.std_wait_parking
        mu_travel, sigma_travel = 0, 0
        vehicles.append(Vehicle(name='v_{}'.format(i), location=location,
                                parking_time_generator=parking_time_generator(mu_parking, sigma_parking, [0, 15]),
                                travel_time_generator=travel_time_generator(mu_travel, sigma_travel, [0, None])))
    return vehicles


def restaurant_generator(n_restaurants, location_file):
    r"""
    Initializes n restaurants with prep time distribution as specified in the
    parameter file and locations as specified in the location file.

    Arguments
    ==========
    n_restaurants (int): Number of restaurants to generate.
    location_file (str): Path to json file containing a list of restaurant locations.

    Returns
    ========
    List of restaurants (Restaurants).

    """
    restaurants = []
    with open(location_file, 'r') as f:
        location_list = json.load(f)
    for i in range(n_restaurants):
        location = np.array(location_list[i])
        mu, sigma = param.mu_prep_time, param.sigma_prep_time
        mean = np.exp(np.log(mu) + (np.log(sigma) ** 2) / 2)
        restaurants.append(Restaurant(name='r_{}'.format(i),
                                      location=location,
                                      prep_time_generator=prep_time_generator(np.log(mu),
                                                                              np.log(sigma),
                                                                              [0, None]),
                                      avg_prep_time=mean))
    return restaurants


def customer_generator(n_customer, restaurants, dispatcher, current_time, spatial_dist):
    r"""
    Generates a list of customers. The name format for customers is "c_i" where i is the customer index.

    Arguments
    ==========
        n_customer (int): Number of customers to generate.
        restaurants (List): List of all restaurants in simulation.
        dispatcher (Dispatcher): Dispatcher that takes the order.
        current_time (int): Current time step in minutes.
        spatial_dist (callable): Spatial distribution to sample from.

    Returns
    ========
        List of customers (Customer).

    """
    customer_list = []
    for i in range(n_customer):
        global customer_index
        customer_index += 1
        location = spatial_dist()
        customer_list.append(Customer(name='c_{}'.format(customer_index),
                                      location=location,
                                      favorite_restaurants=np.random.choice(restaurants, replace=False,
                                                                            size=param.n_fav_restaurants,).tolist(),
                                      time_constraint=int(np.clip(np.random.normal(loc=param.time_constraint,
                                                                                   scale=param.time_constraint_std,
                                                                                   size=None),
                                                                  a_min=0, a_max=None) + 0.5),
                                      dispatcher=dispatcher,
                                      order_time=current_time))
    return customer_list


def prep_time_generator(mean, sigma, clip_interval):
    r"""
    Returns a log-normal cdf to sample food processing times from.

    Arguments
    ==========
        mean (float): Mean of log-normal cdf.
        sigma (float): Sigma of log-normal cdf.
        clip_interval (Tuple): Lower and upper bound to clip at.

    Returns
    ========
        Callable log-normal cdf with the given parameters.

    """
    def f():
        processing_time = np.clip(np.random.lognormal(mean=mean, sigma=sigma, size=None),
                                  a_min=clip_interval[0], a_max=clip_interval[1])
        return processing_time

    return f


def parking_time_generator(mean, std, clip_interval):
    r"""
    Returns a normal cdf to sample parking times from.

    Arguments
    ==========
        mean (float): Mean of normal cdf.
        std (float): Standard deviation of normal cdf
        clip_interval (Tuple): Lower and upper bound to clip at.

    Returns
    ========
        Callable normal cdf with the given parameters.

    """
    def f():
        parking_time = np.clip(np.random.normal(loc=mean, scale=std, size=None),
                               a_min=clip_interval[0], a_max=clip_interval[1])
        return parking_time

    return f


def raw_travel_time(velocity, origin, destination):
    r"""
    Calculates the travel time between origin and destination based on the velocity.

    Arguments
    ==========
        velocity (float): The vehicle's velocity.
        origin (2D array): Point of origin.
        destination (2D array): Point of destination.

    Returns
    ========
        Travel time in minutes.

    """
    if not isinstance(origin, np.ndarray):
        origin = np.array(origin)
    if not isinstance(destination, np.ndarray):
        destination = np.array(destination)
    path = origin - destination
    distance = np.linalg.norm(path)
    return distance / velocity


def travel_time_generator(mean, std, clip_interval):
    r"""
    Returns a travel time distribution based on raw travel times and a Gaussian perturbation.

    Arguments
    ==========
        mean (float): Mean of normal cdf.
        std (float): Standard deviation of normal cdf.
        clip_interval (Tuple): Lower and upper bound to clip at.

    Returns
    ========
        Callable travel time cdf with the given parameters.

    """
    def f(velocity, origin, destination):
        raw_time = raw_travel_time(velocity=velocity, origin=origin, destination=destination)
        delay_time = np.clip(np.random.normal(loc=mean, scale=std, size=None),
                             a_min=clip_interval[0], a_max=clip_interval[1])
        return raw_time + delay_time
    return f


class Iowa:
    r""""
    Spatial distribution of customer demand in Iowa city.
    """

    def __init__(self, file):
        r"""
           Initialize the vehicle.

           Params
           =======
               file (string):  Path to json file containing list of customer locations to sample from.

           Attributes
           ===========
               locations (array): Array of all customer locations.

           """
        self.file = file
        with open("instances/iowa/iowa_customer_locations.json", 'r') as f:
            self.locations = np.array(json.load(f))

    def iowa_distribution(self, n=1):
        r"""
        Sample n=1 customers from Iowa city.

        Arguments
        ==========
            n (int): Number of customer locations to sample.

        Returns
        ========
            Array of customer locations

        """
        indices = np.random.choice(np.arange(len(self.locations)), size=n, replace=False)
        locations = self.locations[indices]
        if n == 1:
            return locations[0]
        else:
            return np.array(locations)
