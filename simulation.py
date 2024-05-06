import numpy as np
import random
import datetime as dt 
import csv 
import pandas as pd
import json
import os

def choose_chargertype(delay_wgt,c_fast,c_slow,L_fast,L_slow,mu_fast,mu_slow,servetime_fast,servetime_slow,p_fast,p_slow):
    #For a given customer and set of conditions about queue lengths, decide which charger type to use
    #Run every time a customer arrives
    #Input: delay_wgt: the weight this customer applies to the delay
            #c_fast,c_slow: number of chargers at each station type
            #L_fast, L_slow: current queue lengths, INCLUDING ppl currently charging, at fast and slow stations
            #mu_fast, mu_slow: average service TIME (not rate) at each station (in minutes)
            #servetime_fast, servetime_slow: this customer's service time at each station (in minutes)
            #p_fast,p_slow: price per minute at the fast and slow stations
    #Output: a string "fast" if we choose the fast station, "slow" if we choose the slow station
            #est_cost_fast, est_cost_slow: the customer's cost estimate for each charger type
    
    #find estimated waits (in minutes)
    if L_fast>=c_fast:
        W_fast=(mu_fast*(L_fast-c_fast+1))/c_fast + servetime_fast #Bonnie: Why +1 here? Because only 1 person needs to leave for us to get service?
    else: #if no wait
        W_fast=servetime_fast
    if L_slow>=c_slow:
        W_slow=(mu_slow*(L_slow-c_slow+1))/c_slow + servetime_slow
    else: #if no wait
        W_slow=servetime_slow

    #calculate total costs
    est_cost_fast=p_fast*servetime_fast + delay_wgt*W_fast
    est_cost_slow=p_slow*servetime_slow + delay_wgt*W_slow

    #pick the lower
    if est_cost_fast < est_cost_slow:
        return "fast", est_cost_fast, est_cost_slow
    else:
        return "slow", est_cost_fast, est_cost_slow
    
def generate_cust(lamb, mu_fast, mu_slow, mean_wgt_ds, mean_wgt_ps, \
                  std_dev_wgt_ds, std_dev_wgt_ps, num_customers=1, pct_ds=0.5):
    #generate a set of customer parameters for each arriving customer.
    #modified to generate everything for multiple customers (for speed purposes)
    #Input: lamb: total arrival rate for customers into the system
            #mu_fast, mu_slow: average service TIME (not rate) at each station
            #mean_wgt_ds, mean_wgt_ps: average values for the delay weight for delay- and price- sensitive customers
            #std_dev_wgt_ds, std_dev_wgt_ps: standard deviations for the delay weight
            #pct_ds: percent of the people arriving who are sampled from the delay sensitive distribution, preemptively set to 0.5
    #Output: delay_wgt: the weight applied to a customer's delay, from a bimodal normal dist
            #servetime_fast, servetime_slow: the amount of time a customer would spend at the fast or slow station
    #TODO we need to pick actually reasonable values for delay_wgt so that it's on the same "scale" as price
            #after being multiplied by the wait time
        #right now it is drawn from one of two normal distributions (50% chance of being from each) with mean
            #values mean_wgt_ds, mean_wgt_ps. Then I bump it to 0 if it's negative
        #but finding the correct values of mean_wgt_ds, mean_wgt_ps is important

    #Customer interarrival times
    #For arrival rate lamb, numpy needs input 1/lamb (see documentation). Same as Exp(lamb), implied by Poisson arrivals
    interarrival_times = np.random.exponential(scale=(1/lamb), size=num_customers) 

    arrival_time = 0
    arrival_times = []
    for interarrival in interarrival_times:
        arrival_time+= interarrival
        arrival_times.append(arrival_time)
    
    cts=np.random.rand(num_customers)
    delay_wgts = []

    #Stores which distribution they were sampled from
    #In analysis, could simply make a cutoff for "delay sensitive" customers based on the delay weight for each customer
    #(rather than using the originating distribution for sensitivity)
    sens_types = [] #'delay' or 'price' (sensitive)

    for ct in cts:
        if ct<pct_ds:
            delay_wgts.append(max(np.random.normal(mean_wgt_ds, std_dev_wgt_ds), 0))
            sens_types.append('delay')
        else:
            delay_wgts.append(max(np.random.normal(mean_wgt_ps, std_dev_wgt_ps), 0))
            sens_types.append('price')

    #old code
    # ct=np.random.rand()
    # if ct<pct_ds:
    #     delay_wgt=np.random.normal(mean_wgt_ds, mean_wgt_ps/2)
    # else:
    #     delay_wgt=np.random.normal(mean_wgt_ps, mean_wgt_ps/2)
    # if delay_wgt<0:
    #     delay_wgt=0

    #only randomly generate one, for fairness
    servetimes_slow=np.random.exponential(scale= mu_slow, size= num_customers)
    servetimes_fast=servetimes_slow*(mu_fast/mu_slow) #Equivalent to random.exponential(1/mu_fast)

    return arrival_times, interarrival_times, delay_wgts, sens_types, servetimes_fast,servetimes_slow

#old code
# def arrival(L_slow,L_fast,mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps,c_fast,c_slow,p_fast,p_slow, pct_ds):
#     #every time a customer arrives. Takes in the current state information, as well as constant params
#     #Input: #L_slow,L_fast: the queue lengths (INCLUDING ppl charging) at time arr_time
#             #mu_fast, mu_slow, mean_wgt_ds, mean_wgt_ps, pct_ds: constant params, defined in generate_cust()
#             #c_fast,c_slow: num chargers of each type
#             #p_fast,p_slow: price of each type. "constant" within each simulation, but we manually change between them
#     #Output:L_slow,L_fast: new queue lengths after customer chooses a queue

#     delay_wgt,servetime_fast,servetime_slow=generate_cust(mu_fast, mu_slow, mean_wgt_ds, mean_wgt_ps, pct_ds)

#     charger_choice=choose_chargertype(delay_wgt,c_fast,c_slow,L_fast,L_slow,mu_fast,mu_slow,servetime_fast,servetime_slow,p_fast,p_slow)
    
#     if charger_choice=="fast":
#         L_fast=L_fast+1
#     else:
#         L_slow=L_slow+1
    
#     #TODO: ALL this does rn is updates queue lengths. But also every person needs a total cost calculated. Use this method only if it's helpful, ignore otherwise
#     return L_fast,L_slow

class Customer:
    
    #A record representing a single customer in the system 
    #The init function just sets default values for every field, and then we can incrementally set them in the simulation 
    #Stores fields we will need for system analysis for all customers. 
    #If there's a customer-specific operation we need to do, we can define it as a method in this class and then apply at scale

    def __init__(self, arrival_time=None, interarrival_time=None, departure_time=None, delay_wgt=None, sens_type = None, servetime_fast=None,\
                servetime_slow=None, L_slow=None, L_fast=None, charger_choice = None, est_cost_fast = None, est_cost_slow = None,\
                p_fast=None, p_slow = None):

        #Customer's arrival into their respective queue
        self.arrival_time = arrival_time

        #Customer's time to arrive after the previous customer (or after simulation start)
        self.interarrival_time = interarrival_time

        #Customer's departure time
        self.departure_time = departure_time

        #Customer's delay sensitivity
        self.delay_wgt = delay_wgt

        #Distribution customer's delay sensitivity was sampled from, we may never use this but I include it anyway
        self.sens_type = sens_type

        #Customer's service time at the fast chargers
        self.servetime_fast = servetime_fast

        #Customer's service time at the slow chargers
        self.servetime_slow = servetime_slow

        #Queue length (including people at chargers) for the fast chargers when this customer arrives
        self.L_fast = L_fast

        #Queue length (including people at chargers) for the slow chargers when this customer arrives
        self.L_slow = L_slow

        #Customer's charger choice
        self.charger_choice = charger_choice

        #Customer's estimate of their cost for fast chargers
        self.est_cost_fast = est_cost_fast

        #Customer's estimate of their cost for slow chargers
        self.est_cost_slow = est_cost_slow

        #Prices for each type of charger by minute
        self.p_fast = p_fast
        self.p_slow = p_slow

        #Fields defined in the future by calling methods, requires the other fields
        self.true_cost = None #true full cost customer incurs
        self.true_total_wait = None #true customer wait time (not their estimate)
        self.true_queue_wait = None #time customer waited in system without service
        self.true_servetime = None #actual service time of customer (for charger they chose)

    def compute_true_cost(self):
        #The true cost that the customer incurs from their choice. 
        #Stores the result in self.true_cost and returns true_cost.
        #Populates true_total_wait and true_queue_wait, the total system wait time and queue wait time for the customer
        #Call this method once the necessary fields have been populated for this customer by the simulation.
        #Only needs to do the computation if we haven't previously computed this field 

        if self.true_cost == None:
            true_wait = self.departure_time - self.arrival_time
            self.true_total_wait = true_wait

            if self.charger_choice == 'fast':

                self.true_cost = self.p_fast*self.servetime_fast + self.delay_wgt*(true_wait)
                self.true_queue_wait = true_wait - self.servetime_fast
                self.true_servetime = self.servetime_fast

            else:

                self.true_cost = self.p_slow*self.servetime_slow + self.delay_wgt*(true_wait)
                self.true_queue_wait = true_wait - self.servetime_slow
                self.true_servetime = self.servetime_slow

        return self.true_cost


def simulate(num_customers,p_fast,p_slow,mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps, std_dev_wgt_ds, std_dev_wgt_ps, c_fast,c_slow,lamb, pct_ds,\
             log_events = False, random_seed=None):
    #run the simulation

    #Input: #num_customers:number of customers to arrive
            #p_fast,p_slow: price PER MINUTE of fast and slow chargers, respectively. Change each simulation
            #mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps,c_fast,c_slow, pct_ds: constant params, defined where they are used
            #lamb: arrival rate into the system at large. Should pick based on the c and mu values to not blow up the system
            #log_events: if to return the chronological log of events in the system (for debugging). Otherwise return empty event log.
            #            Failure events are always logged regardless of input.
            #random_seed: if we want to specify a random seed, so that we can always get the same randomness under different params.

    #Output: total_cost: the total cost incurred by all customers during the simulation 
           # customer_results: information about every customer's individual parameters, cost, and wait times in the simulation
    
    #We will need to use only numpy random number generators to follow the same random seed
    if random_seed != None:
        np.random.seed(random_seed)

    #log of all events
    event_log = []

    #Get all random customer properties that can be computed for each cust. without knowing anything about other customers.
    arrival_times, interarrival_times, delay_wgts, sens_types, servetimes_fast, servetimes_slow = generate_cust(lamb,\
                                                                                                                mu_fast,\
                                                                                                                mu_slow,\
                                                                                                                mean_wgt_ds,\
                                                                                                                mean_wgt_ps,\
                                                                                                                std_dev_wgt_ds,\
                                                                                                                std_dev_wgt_ps,\
                                                                                                                num_customers,\
                                                                                                                pct_ds)


    #Use outputs from generate_cust to create simulation customers

    #Populate what we know into about each customer 
    #List of customers with arrival times, delay sensitivty information, service times for each charger
    #Ordered chronologically by arrival time
    #Need to populate with information about their true wait times and departure during the simulation
    customer_results = [Customer(arrival_time=arrival_times[i], interarrival_time=interarrival_times[i], delay_wgt=delay_wgts[i],\
                                sens_type=sens_types[i], servetime_fast=servetimes_fast[i], servetime_slow=servetimes_slow[i],\
                                p_fast=p_fast, p_slow=p_slow) for i in range(num_customers)] 

    #initial values for L_slow and L_fast (queue lengths). I have code to increase when someone arrivals but not to decrease when someone leaves
    # L_slow=0
    # L_fast=0

    #Initialize simulation clock
    clock=0 

    #Index of the next customer to arrive
    idx_arriving_cust = 0

    #Queues for customers in the system but not at a server 
    #Queues are ordered - the first person in the list has been there the longest. New members are appended to the end. 
    slow_queue = []
    fast_queue = []

    #Customers currently in service
    fast_servers = [] #max c_fast
    slow_servers = [] #max c_slow

    #give the event name and the associated future clock time when the event occurs
    #events: 'arrival', 'leave fast', 'leave slow'
    upcoming_events = [('arrival', customer_results[idx_arriving_cust].arrival_time)] 

    while len(upcoming_events) >0:
        #Find which event occurs next based on time: someone arrives or someone departs
        #Organize system after each event, should never have someone in a queue if a server is open
        #Update the clock accordingly

        #Sanity check
        if len(fast_servers)>c_fast:
            print('FAILED - too many people fast charging')
            event_log.append(('FAIL - too many fast charging', -1))
            return 0, customer_results, event_log
        
        if len(slow_servers)>c_slow:
            print('FAILED - too many people slow charging')
            event_log.append(('FAIL - too many slow charging', -1))
            return 0, customer_results, event_log

        #find the next occurring event
        event = min(upcoming_events, key=lambda x:x[1]) 

        if log_events:
            event_log.append(event)

        #update the simulator clock to the new event time
        clock = event[1] 

        if event[0] == 'arrival':

            #number of people in system who are queueing for or actively using a type of charger
            L_fast = len(fast_queue) + len(fast_servers)
            L_slow = len(slow_queue) + len(slow_servers)
            
            #customer arrives
            arriving_cust = customer_results[idx_arriving_cust]

            #customer chooses a charger type
            charger_choice, est_cost_fast, est_cost_slow = choose_chargertype(arriving_cust.delay_wgt,c_fast,c_slow,\
                                                L_fast,L_slow,mu_fast,mu_slow,\
                                                arriving_cust.servetime_fast,arriving_cust.servetime_slow,\
                                                p_fast,p_slow)
            
            #record customer information
            arriving_cust.charger_choice = charger_choice
            arriving_cust.est_cost_fast = est_cost_fast
            arriving_cust.est_cost_slow = est_cost_slow
            arriving_cust.L_fast = L_fast
            arriving_cust.L_slow = L_slow


            #if not all chargers are full, serve immediately 
            #and add their departure event to the list of events we have to consider 
            #else, add them to the charger queue

            if charger_choice == 'fast':
                if len(fast_servers)< c_fast:
                    #not all chargers full
                    fast_servers.append(arriving_cust)
                    #plan to leave at the time service completes
                    departure_time = clock + arriving_cust.servetime_fast
                    arriving_cust.departure_time = departure_time #record so we know which customer to remove at this time
                    upcoming_events.append(('leave fast', departure_time))
                    
                    if log_events:
                        #log extra event - for metadata purposes
                        event_log.append(('served by fast', clock))
                else:
                    #all chargers full
                    fast_queue.append(arriving_cust)

                    if log_events:
                        #log extra event - for metadata purposes
                        event_log.append(('waiting for fast', clock))

            elif charger_choice == 'slow':
                if len(slow_servers)< c_slow:
                    #not all chargers full
                    slow_servers.append(arriving_cust)
                    #plan to leave at the time service completes
                    departure_time = clock + arriving_cust.servetime_slow
                    arriving_cust.departure_time = departure_time #record so we know which customer to remove at this time
                    upcoming_events.append(('leave slow', departure_time))

                    if log_events:
                        #log extra event - for metadata purposes
                        event_log.append(('served by slow', clock))
                    
                else:
                    #all chargers full
                    slow_queue.append(arriving_cust)

                    if log_events:
                        #log extra event - for metadata purposes
                        event_log.append(('waiting for slow', clock))

            else:
                print("FAILED - Invalid charger type")
                event_log.append(('FAIL - invalid charger type', -1))
                return 0, customer_results, event_log

            #Start watching for the next arrival, if there is one
            idx_arriving_cust+= 1
            if idx_arriving_cust< len(customer_results):
                next_customer = customer_results[idx_arriving_cust]
                upcoming_events.append(('arrival', next_customer.arrival_time))
                #Note: could have computed arrival time here, rather than before

            else:
                if log_events:
                    #log extra event - for metadata purposes
                    event_log.append((f'arrivals completed - {idx_arriving_cust}', clock))
            

        elif event[0] == 'leave fast':

            #have customer depart 
            #order customers by departure time, the first to depart should leave
            fast_servers.sort(key = lambda x: x.departure_time) 

            if fast_servers[0].departure_time != event[1]:
                #sanity check
                print("FAILED - wrong customer departing fast queue")
                event_log.append(('FAIL - wrong customer departing fast queue', -1))
                return 0, customer_results, event_log

            fast_servers = fast_servers[1:]

            #if anyone is waiting to be served
            if len(fast_queue)>0:
                #serve the next customer - the oldest person at the front of the queue 
                serving_customer = fast_queue[0]

                #remove from queue and serve
                fast_queue = fast_queue[1:]
                fast_servers.append(serving_customer)
                departure_time = clock + serving_customer.servetime_fast
                serving_customer.departure_time = departure_time

                #add their departure event to the upcoming events 
                upcoming_events.append(('leave fast', departure_time))

                if log_events:
                    #log extra event - for metadata purposes
                    event_log.append(('move cust. from queue to fast server', clock))


        elif event[0] == 'leave slow':
            #have customer depart 
            #order customers by departure time, the first to depart should leave
            slow_servers.sort(key = lambda x: x.departure_time) 

            if slow_servers[0].departure_time != event[1]:
                #sanity check
                print("FAILED - wrong customer departing slow queue")
                event_log.append(('FAIL - wrong customer departing slow queue', -1))
                return 0, customer_results, event_log

            slow_servers = slow_servers[1:]

            #if anyone is waiting to be served
            if len(slow_queue)>0:
                #serve the next customer - the oldest person at the front of the queue 
                serving_customer = slow_queue[0]

                #remove from queue and serve
                slow_queue = slow_queue[1:]
                slow_servers.append(serving_customer)
                departure_time = clock + serving_customer.servetime_slow
                serving_customer.departure_time = departure_time

                #add their departure event to the upcoming events 
                upcoming_events.append(('leave slow', departure_time))

                if log_events:
                    #log extra event - for metadata purposes
                    event_log.append(('move cust. from queue to slow server', clock))

        else:
            print("FAILED - Invalid event")
            event_log.append(('FAIL - invalid event', -1))
            return 0, customer_results, event_log
        
        #Removed processed event from upcoming events
        upcoming_events.remove(event)

    #Sum of realized costs for all customers, also computes true wait times
    total_cost = np.sum([customer.compute_true_cost() for customer in customer_results])
    event_log.append(('Finished - Success', 1))

    return total_cost, customer_results, event_log

#Functions to save our results from a run 

def make_total_cost_file(total_cost, filename):
    with open(f'{filename}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([total_cost])

def customer_to_dict(customer):
    
    customer_dict = { 'arrival_time': customer.arrival_time,\
                      'interarrival_time': customer.interarrival_time,\
                      'departure_time': customer.departure_time,\
                      'delay_wgt': customer.delay_wgt,\
                      'sens_type': customer.sens_type,\
                      'servetime_fast': customer.servetime_fast,\
                      'servetime_slow': customer.servetime_slow,\
                      'L_fast': customer.L_fast,\
                      'L_slow': customer.L_slow,\
                      'charger_choice': customer.charger_choice,\
                      'est_cost_fast': customer.est_cost_fast,\
                      'est_cost_slow': customer.est_cost_slow,\
                      'p_fast': customer.p_fast,\
                      'p_slow': customer.p_slow,\
                      'true_cost': customer.true_cost,\
                      'true_total_wait': customer.true_total_wait,\
                      'true_queue_wait':customer.true_queue_wait,\
                      'true_servetime': customer.true_servetime}

    return customer_dict

def make_customer_results_file(customer_results, filename):
    #File of customers and their decisions, the order of the customers in the CSV is the order they appear in the simulation

    customer_results_dicts = [customer_to_dict(customer) for customer in customer_results]
    pd.DataFrame(customer_results_dicts).to_csv(f'{filename}.csv')

def make_event_log_file(event_log, filename):
    events = [list(event) for event in event_log]

    with open(f'{filename}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(events)

def make_param_file(param_dict, filename):
    with open(f'{filename}.json', 'w') as outfile: 
        json.dump(param_dict, outfile)


if __name__ == "__main__":
    #input params (do NOT change these between trials)
    #TODO change these obviously from what they are now

    num_runs = 10 #Number of simulation runs

    num_customers=20

    description = "Simulation run"
    mu_fast=30
    mu_slow=90
    mean_wgt_ds=10
    mean_wgt_ps=2

    #both have the same std. dev for now
    std_dev_wgt_ds = mean_wgt_ps/2
    std_dev_wgt_ps = mean_wgt_ps/2

    c_fast=2
    c_slow=2
    lamb= 0.05
    pct_ds=0.5

    #prices (change these between trials)
    p_fast=0.5
    p_slow=0

    #random seeds, if any (can be None)
    #up to 20 here --> if num_runs=10, we will only use the first 10
    random_seeds = [975, 898, 389, 672, 89, 470, 970, 17, 340, \
                    925, 712, 609, 481, 385, 883, 833, 911, 264, 872, 553]

    #whether to log events (failure events always logged regardless of parameter)
    log_events = True

    #unique string for filenames, the time at which the results were generated for this run
    date_str = dt.datetime.now().strftime('%b%-d-%I-%M-%-S%p')

    #make file to store runs
    results_folder = f'results/results_{date_str}'
    os.mkdir(results_folder)

    param_dict = {'num_customers': num_customers,\
                  'mu_fast': mu_fast,\
                  'mu_slow': mu_slow,\
                  'mean_wgt_ds': mean_wgt_ds,\
                  'mean_wgt_ps': mean_wgt_ps,\
                  'std_dev_wgt_ds': std_dev_wgt_ds,\
                  'std_dev_wgt_ps': std_dev_wgt_ps,\
                  'c_fast': c_fast,\
                  'c_slow': c_slow,\
                  'lamb': lamb,\
                  'pct_ds': pct_ds,\
                  'p_fast': p_fast,\
                  'p_slow': p_slow,\
                  'random_seed': random_seeds,\
                  'log_events': log_events, 
                  'description': description}
    
    #save run params
    make_param_file(param_dict, f'{results_folder}/params_{date_str}')

    for run in range(num_runs):
        run_str = f'run{run}'

        total_cost, customer_results, event_log = simulate(num_customers,p_fast,p_slow,mu_fast,mu_slow,\
                                                        mean_wgt_ds,mean_wgt_ps,\
                                                        std_dev_wgt_ds, std_dev_wgt_ps,\
                                                            c_fast,c_slow,lamb,\
                                                            pct_ds, log_events, random_seeds[run])

        print(f'{run_str}: {total_cost}')

        #Save all results


        make_total_cost_file(total_cost, f'{results_folder}/total_cost_{date_str}_{run_str}')
        make_customer_results_file(customer_results, f'{results_folder}/customer_results_{date_str}_{run_str}')
        make_event_log_file(event_log, f'{results_folder}/event_log_{date_str}_{run_str}')