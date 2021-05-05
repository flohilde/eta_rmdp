import parameters as param
from etd import get_all_restaurants_etd, advanced_eta
from dispatcher import Dispatcher
from generator import restaurant_generator, vehicle_generator, customer_generator, Iowa
from simulation import simulate, approx_simulate
import numpy as np
from datetime import datetime
import simplejson as json
import torch
import timeit
import lightgbm as lgb
import matplotlib.pyplot as plt
import deep_insertion

if __name__ == '__main__':

    # set seed
    np.random.seed(0)

    """Initialization"""
    # initialize restaurants
    restaurants = restaurant_generator(n_restaurants=param.n_restaurants,
                                       location_file="instances/iowa/iowa_restaurant_locations.json")
    restaurant_dict = {restaurant.name: restaurant for restaurant in restaurants}

    # initialize vehicles
    vehicles = vehicle_generator(n_vehicles=param.n_vehicles)

    # initialize dispatcher
    dispatcher = Dispatcher(vehicles=vehicles, restaurants=restaurants)

    # intitalize customer locations
    iowa = Iowa(file="instances/iowa/iowa_customer_locations.json")

    # initialize customer list
    customers = []

    # data to save
    customers_output = []
    time_per_customer = []

    if param.n_approx_simulations is not None and param.n_approx_simulations > 0:
        model = deep_insertion.DeepInsertionModel(encoder_shape=65, fc1_shapes=[128, 128, 128, 128, 11],
                                                  fc2_shapes=[128, 128, 128, 128, 12])
        model.load_state_dict(torch.load("models/dnn/deep_insertion.pt"))

    if advanced_eta:
        gbm = lgb.Booster(model_file='models/gbdt/gbdt.txt')

    """Loop"""
    # Loop over days
    for d in range(param.n_days):
        print("Day {}.".format(d))
        # sample order times:
        n_lunch = int(max(0, np.random.normal(loc=param.n_lunch_mu, scale=param.n_lunch_sigma)))
        n_dinner = int(max(0, np.random.normal(loc=param.n_dinner_mu, scale=param.n_dinner_sigma)))
        lunch = np.random.normal(loc=param.t_lunch_mu, scale=param.t_lunch_sigma, size=n_lunch)
        dinner = np.random.normal(loc=param.t_dinner_mu, scale=param.t_dinner_sigma, size=n_dinner)
        order_times = np.sort(np.hstack((lunch, dinner)))
        order_times = order_times[(0 < order_times) & (order_times < 1439)].tolist()

        # Loop over one day (1 minute steps)
        for t in range(param.start_time_day + 1, param.end_time_day + 1):
            """New Event"""
            # determine number of orders
            n_customer = 0
            while len(order_times) > 0 and ((t - 1) < order_times[0] <= t):
                n_customer += 1
                order_times.pop(0)
            # generate customers accordingly
            new_customers = customer_generator(n_customer=n_customer,
                                               restaurants=restaurants,
                                               dispatcher=dispatcher,
                                               current_time=t,
                                               spatial_dist=iowa.iowa_distribution)

            # customers place orders
            for customer in new_customers:
                start = timeit.default_timer()
                # calculate etd for every customer-restaurant tuple
                etd_dict = get_all_restaurants_etd(customer=customer, dispatcher=dispatcher,
                                                   restaurants=customer.favorite_restaurants, current_time=t)
                if advanced_eta:
                    etd_dict = advanced_eta(etd_dict=etd_dict, restaurant_dict=restaurant_dict,
                                            order_time=t, model=gbm)

                # place an order and update routes accordingly
                customer.status = customer.order_restaurant(etd_dict=etd_dict, current_time=t)
                customers.append(customer)
                print("c_{}".format(len(customers)))
                # start simulation
                if param.n_simulations is not None and param.n_simulations > 0 and customer.status == 0:
                    # create a state_dict
                    sim_customers = [customer for customer in customers if customer.status == 0]
                    state_dict = {'t': t,
                                  'customer': customer.name,
                                  'customers': sim_customers,
                                  'vehicles': vehicles,
                                  'restaurants': restaurants}
                    atds = simulate(state_dict, spatial_distrib=iowa.iowa_distribution,
                                    nodes=param.simulation_nodes)
                    customer.simulated_etd = atds

                if param.n_approx_simulations is not None and param.n_approx_simulations > 0 and customer.status == 0:
                    # create a state_dict
                    sim_customers = [customer for customer in customers if customer.status == 0]
                    state_dict = {'t': t,
                                  'customer': customer.name,
                                  'customers': sim_customers,
                                  'vehicles': vehicles,
                                  'restaurants': restaurants}
                    atds = approx_simulate(state_dict, model, iowa.iowa_distribution)
                    customer.simulated_etd = atds
                stop = timeit.default_timer()
                print('Time: ', stop - start)
                time_per_customer.append(stop - start)
                customer.day = d

            """Complete Actions"""
            # Restaurant actions
            for restaurant in restaurants:
                restaurant.check_queue(t)
            # Vehicle actions
            for vehicle in vehicles:
                vehicle.act(current_time=t)

            """Visualize"""
            if param.visualize:

                if t % 5 == 0 and (10 * 60 < t < 14 * 60 or 16 * 60 < t < 20 * 60) or t == 1440:

                    # position of restaurants and customers
                    restaurant_loc = np.array([restaurant.location for restaurant in restaurants])
                    waiting_customers = np.array([customer for customer in customers if customer.status == 0])
                    served_customers = np.array([customer for customer in customers if customer.status == 1])
                    rejected_customers = np.array([customer for customer in customers if customer.status == -1])
                    w_customer_loc = np.array([customer.location for customer in waiting_customers])
                    etd_customer = ["%d:%02d" % (customer.etd // 60, customer.etd % 60)
                                    for customer in waiting_customers]
                    s_customer_loc = np.array([customer.location for customer in served_customers])
                    diff_atd_etd_customer = [int(customer.atd - customer.etd) for customer in served_customers]
                    r_customer_loc = np.array([customer.location for customer in rejected_customers])

                    # position, absolved and planned routes of vehicles
                    vehicle_loc = np.array([vehicle.location for vehicle in vehicles])
                    route_list = [[vehicle.location] for vehicle in vehicles]
                    loc_log = []
                    k = 0
                    for vehicle in vehicles:
                        for action in vehicle.route:
                            if action.status in [1, 2]:
                                route_list[k].append(action.destination)
                        k += 1

                    plt.scatter(restaurant_loc[:, 0], restaurant_loc[:, 1], c='b', marker='s', label='Restaurant')
                    if len(w_customer_loc) != 0:
                        plt.scatter(w_customer_loc[:, 0], w_customer_loc[:, 1], c='k',
                                    marker='.', s=100, label='Waiting Customer')
                        # add etd for customer
                        for i, txt in enumerate(etd_customer):
                            plt.annotate(txt, (w_customer_loc[i][0] + 0.01, w_customer_loc[i][1] + 0.01))
                    if len(r_customer_loc) != 0:
                        plt.scatter(r_customer_loc[:, 0], r_customer_loc[:, 1], c='r',
                                    marker='.', s=100, label='Rejected Customer')
                    if len(s_customer_loc) != 0:
                        plt.scatter(s_customer_loc[:, 0], s_customer_loc[:, 1], c='g',
                                    marker='.', s=100, label='Served Customer')
                        for i, txt in enumerate(diff_atd_etd_customer):
                            plt.annotate(txt, (s_customer_loc[i][0] + 0.01, s_customer_loc[i][1] + 0.01))
                    route_colors = ['k', 'r', 'g', 'b', 'c'] * 20
                    for color_index in range(len(vehicle_loc)):
                        plt.scatter(vehicle_loc[color_index, 0], vehicle_loc[color_index, 1],
                                    c=route_colors[color_index], marker='>', label='Vehicle', s=200, alpha=0.5)
                    for color_index, route in enumerate(route_list):
                        route = np.array(route)
                        plt.plot(route[:, 0], route[:, 1], ls='--', c=route_colors[color_index])
                    plt.title("%d:%02d" % (t // 60, t % 60))
                    plt.xlim((0, 16))
                    plt.ylim((0, 12))
                    plt.show()
                    """
                    
        """"Save output"""
        if param.write_data:
            customers_output.extend([customer.to_dict() for customer in customers])

        """Reset for next day"""
        customers = []
        for vehicle in vehicles:
            vehicle.route = []
            vehicle.action_time = 0
            vehicle.started_action_time = 0
            vehicle.location_log = [vehicle.location]

    if param.write_data:
        now = datetime.now()
        file = param.write_to + now.strftime("%m_%d_%Y_%H_%M_%S") + ".json"

        with open(file, 'w') as f:
            json.dump(customers_output, f)
