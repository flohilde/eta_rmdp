import numpy as np
import parameters as param
from generator import customer_generator
from etd import get_all_restaurants_etd
from dispatcher import Dispatcher, FakeDispatcher
from pathos.multiprocessing import ProcessingPool as Pool


def simulate(state_dict, spatial_distrib, nodes=None):
    r"""
    Full online simulation of the delivery process.

    Arguments
    ==========
    state_dict (Dict): Dictionary containing all relevant information for the simulation.
    spatial_distrib (callable): Spatial distribution to sample customer locations from.
    nodes (int): Number of nodes to simulate on in parallel.

    Returns
    ========
    Returns a list of simulated arrival times for a given (customer, restaurant) Tuple.

    """
    # start simulation
    if nodes is None or nodes == 0:
        atd_list = []
        for i in range(param.n_simulations):
            # load copy of state
            t = state_dict["t"]
            restaurants = [restaurant.full_deepcopy() for restaurant in state_dict["restaurants"]]
            restaurant_dict = {}
            for restaurant in restaurants:
                restaurant_dict[restaurant.name] = restaurant
            customers = [customer.full_deepcopy() for customer in state_dict["customers"]]
            customer_dict = {}
            for customer in customers:
                customer_dict[customer.name] = customer
            customer_sim = customer_dict[state_dict["customer"]]
            # Repopulate vehicles
            vehicles = [vehicle.full_deepcopy(restaurant_dict=restaurant_dict,
                                              customer_dict=customer_dict)
                        for vehicle in state_dict["vehicles"]]
            dispatcher = Dispatcher(vehicles=vehicles, restaurants=restaurants)
            # create customers
            n_lunch = int(max(0, np.random.normal(loc=param.n_lunch_mu, scale=param.n_lunch_sigma)))
            n_dinner = int(max(0, np.random.normal(loc=param.n_dinner_mu, scale=param.n_dinner_sigma)))
            lunch = np.random.normal(loc=param.t_lunch_mu, scale=param.t_lunch_sigma, size=n_lunch)
            dinner = np.random.normal(loc=param.t_dinner_mu, scale=param.t_dinner_sigma, size=n_dinner)
            order_times = np.sort(np.hstack((lunch, dinner)))
            order_times = order_times[(t-1 < order_times) & (order_times < 1439)].tolist()
            # start loop
            while customer_sim.status == 0:
                # determine number of orders
                n_customer = 0
                while len(order_times) > 0 and ((t - 1) < order_times[0] < t):
                    n_customer += 1
                    order_times.pop(0)
                # generate customers accordingly
                new_customers = customer_generator(n_customer=n_customer,
                                                   restaurants=restaurants,
                                                   dispatcher=dispatcher,
                                                   current_time=t,
                                                   spatial_dist=spatial_distrib)
                # customers place orders
                for customer in new_customers:
                    # calculate etd for every customer-restaurant tuple
                    etd_dict = get_all_restaurants_etd(customer=customer, dispatcher=dispatcher,
                                                       restaurants=customer.favorite_restaurants, current_time=t)
                    # place an order and update routes accordingly
                    customer.status = customer.order_restaurant(etd_dict=etd_dict, current_time=t)

                # Complete Actions
                # Restaurant actions
                for restaurant in restaurants:
                    restaurant.check_queue(t)
                # Vehicle actions
                for vehicle in vehicles:
                    vehicle.act(current_time=t)
                t += 1
            atd_list.append(customer_sim.atd)
        return atd_list

    if nodes is not None:
        def simulation_loop(days, seeds):
            np.random.seed(int(seeds))
            atd_list = []
            for d in days:
                # load copy of state
                t = state_dict["t"]
                restaurants = [restaurant.full_deepcopy() for restaurant in state_dict["restaurants"]]
                restaurant_dict = {}
                for restaurant in restaurants:
                    restaurant_dict[restaurant.name] = restaurant
                customers = [customer.full_deepcopy() for customer in state_dict["customers"]]
                customer_dict = {}
                for customer in customers:
                    customer_dict[customer.name] = customer
                customer_sim = customer_dict[state_dict["customer"]]
                # Repopulate vehicles
                vehicles = [vehicle.full_deepcopy(restaurant_dict=restaurant_dict,
                                                  customer_dict=customer_dict)
                            for vehicle in state_dict["vehicles"]]
                dispatcher = Dispatcher(vehicles=vehicles, restaurants=restaurants)
                # create customers
                n_lunch = int(max(0, np.random.normal(loc=param.n_lunch_mu, scale=param.n_lunch_sigma)))
                n_dinner = int(max(0, np.random.normal(loc=param.n_dinner_mu, scale=param.n_dinner_sigma)))
                lunch = np.random.normal(loc=param.t_lunch_mu, scale=param.t_lunch_sigma, size=n_lunch)
                dinner = np.random.normal(loc=param.t_dinner_mu, scale=param.t_dinner_sigma, size=n_dinner)
                order_times = np.sort(np.hstack((lunch, dinner)))
                order_times = order_times[(t - 1 < order_times) & (order_times < 1439)].tolist()
                # start loop
                while customer_sim.status == 0:
                    # determine number of orders
                    n_customer = 0
                    while len(order_times) > 0 and ((t - 1) < order_times[0] < t):
                        n_customer += 1
                        order_times.pop(0)
                    # generate customers accordingly
                    new_customers = customer_generator(n_customer=n_customer,
                                                       restaurants=restaurants,
                                                       dispatcher=dispatcher,
                                                       current_time=t,
                                                       spatial_dist=spatial_distrib)
                    # customers place orders
                    for customer in new_customers:
                        # calculate etd for every customer-restaurant tuple
                        etd_dict = get_all_restaurants_etd(customer=customer, dispatcher=dispatcher,
                                                           restaurants=customer.favorite_restaurants, current_time=t)
                        # place an order and update routes accordingly
                        customer.status = customer.order_restaurant(etd_dict=etd_dict, current_time=t)

                    """Complete Actions"""
                    # Restaurant actions
                    for restaurant in restaurants:
                        restaurant.check_queue(t)
                    # Vehicle actions
                    for vehicle in vehicles:
                        vehicle.act(current_time=t)
                    t += 1
                atd_list.append(customer_sim.atd)
            return atd_list

        pool = Pool(nodes=nodes)
        n = int(param.n_simulations / nodes)
        chunks = [range(k, k + n) for k in range(nodes)]

        # set seed
        ini_seed = int(state_dict["customer"][2:])
        seed_param = list(range(ini_seed * 11111, ini_seed * 11111 + nodes))
        atd_list = pool.map(simulation_loop, chunks, seed_param)
        atd_list = [item for sublist in atd_list for item in sublist]
        return atd_list


def approx_simulate(state_dict, model, spatial_distribution):
    r"""
    Approximate full online simulation of the delivery process using a DNN.


    Arguments
    ==========
    state_dict (Dict): Dictionary containing all relevant information for the simulation.
    model (torch.nn): DeepInsertion model.
    spatial_distrib (callable): Spatial distribution to sample customer locations from.

    Returns
    ========
    Returns a list of simulated arrival times for a given (customer, restaurant) Tuple.

    """
    # start simulation
    atd_list = []
    for i in range(param.n_approx_simulations):
        # load copy of state
        t = state_dict["t"]
        restaurants = [restaurant.full_deepcopy() for restaurant in state_dict["restaurants"]]
        restaurant_dict = {}
        for restaurant in restaurants:
            restaurant_dict[restaurant.name] = restaurant
        customers = [customer.full_deepcopy() for customer in state_dict["customers"]]
        customer_dict = {}
        for customer in customers:
            customer_dict[customer.name] = customer
        customer_sim = customer_dict[state_dict["customer"]]
        # Repopulate vehicles
        vehicles = [vehicle.full_deepcopy(restaurant_dict=restaurant_dict,
                                          customer_dict=customer_dict)
                    for vehicle in state_dict["vehicles"]]
        dispatcher = FakeDispatcher(vehicles=vehicles, restaurants=restaurants, model=model)
        # create customers
        n_lunch = int(max(0, np.random.normal(loc=param.n_lunch_mu, scale=param.n_lunch_sigma)))
        n_dinner = int(max(0, np.random.normal(loc=param.n_dinner_mu, scale=param.n_dinner_sigma)))
        lunch = np.random.normal(loc=param.t_lunch_mu, scale=param.t_lunch_sigma, size=n_lunch)
        dinner = np.random.normal(loc=param.t_dinner_mu, scale=param.t_dinner_sigma, size=n_dinner)
        order_times = np.sort(np.hstack((lunch, dinner)))
        order_times = order_times[(t-1 < order_times) & (order_times < 1439)].tolist()
        # start loop
        while customer_sim.status == 0:
            # determine number of orders
            n_customer = 0
            while len(order_times) > 0 and ((t - 1) < order_times[0] < t):
                n_customer += 1
                order_times.pop(0)
            # generate customers accordingly
            new_customers = customer_generator(n_customer=n_customer,
                                               restaurants=restaurants,
                                               dispatcher=dispatcher,
                                               current_time=t,
                                               spatial_dist=spatial_distribution)
            # customers place orders
            for customer in new_customers:
                # calculate etd for every customer-restaurant tuple
                etd_dict = get_all_restaurants_etd(customer=customer, dispatcher=dispatcher,
                                                   restaurants=[customer.favorite_restaurants[0]], current_time=t)
                # place an order and update routes accordingly
                customer.status = customer.order_restaurant(etd_dict=etd_dict, current_time=t,
                                                            ignore_time_constraint=True)

            """Complete Actions"""
            # Restaurant actions
            for restaurant in restaurants:
                restaurant.check_queue(t)
            # Vehicle actions
            for vehicle in vehicles:
                vehicle.act(current_time=t)
            t += 1
        atd_list.append(customer_sim.atd)
    return atd_list


def scale(X, x_min=-1, x_max=1):
    r"""
    Scales the input of the DeepInsertion Model.

    Arguments
    ==========
        X (array): Unscaled input.
        x_min (float): Lower bound of scaled input.
        x_max (float): Upper bound of scaled input.

    Returns
    ========
    Array containing scaled input.

    """
    X_min = np.array([0.0, 0.0, 3.52070206, 3.71164996, 0.0, 3.0, 0.0, 0.0, 3.0, -20.0,
                      0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0, 0.0, 0.0, -17.0, 0.0, 0.0, 0.0, 0.0,
                      -19.0, 0.0, 0.0, 0.0, 0.0, -15.0, 0.0, 0.0, 0.0, 0.0, -6.0, 0.0, 0.0, 0.0, 0.0, -7.0,
                      0.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X_max = np.array([15.35130731, 13.49346697, 12.46919309, 10.34123165, 51.0, 4.0, 15.35130731,
                      13.49346697, 35.0, 15.0, 4.0, 15.35130731, 13.49346697, 43.0, 15.0, 4.0, 15.35130731,
                      13.49346697, 32.0, 15.0, 4.0, 15.35130731, 13.49346697, 32.0, 15.0, 4.0, 15.35130731,
                      13.49346697, 34.0, 15.0, 4.0, 15.35130731, 13.49346697, 29.0, 15.0, 4.0, 15.35098815,
                      13.49346697, 28.0, 15.0, 4.0, 15.34983257, 13.41473213, 24.0, 15.0, 4.0, 15.33966073,
                      13.49346697, 21.0, 15.0, 4.0, 15.32294646, 12.45623837, 17.0, 15.0, 4.0, 15.15503898,
                      11.43302948, 14.0, 14.0, 4.0, 10.06925151, 10.76896486, 10.0, 12.0])

    nom = (X-X_min)*(x_max-x_min)
    denom = X_max - X_min
    denom[denom == 0] = 1
    return x_min + nom/denom
