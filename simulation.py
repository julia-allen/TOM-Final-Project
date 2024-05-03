import numpy as np
import random

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
    
    #find estimated waits (in minutes)
    if L_fast>=c_fast:
        W_fast=(mu_fast*(L_fast-c_fast+1))/c_fast + servetime_fast
    else: #if no wait
        W_fast=servetime_fast
    if L_slow>=c_slow:
        W_slow=(mu_slow*(L_slow-c_slow+1))/c_slow + servetime_slow
    else: #if no wait
        W_slow=servetime_slow

    #calculate total costs
    totalcost_fast=p_fast*servetime_fast + delay_wgt*W_fast
    totalcost_slow=p_slow*servetime_slow + delay_wgt*W_slow

    #pick the lower
    if totalcost_fast<totalcost_slow:
        return "fast"
    else:
        return "slow"
    
def generate_cust(mu_fast, mu_slow, mean_wgt_ds, mean_wgt_ps):
    #generate a set of customer parameters. Run whenever a customer arrives
    #Input: average service TIME (not rate) at each station
            #mean_wgt_ds, mean_wgt_ps: average values for the delay weight for delay- and price- sensitive customers
    #Output: delay_wgt: the weight applied to a customer's delay, from a bimodal normal dist
            #servetime_fast, servetime_slow: the amount of time a customer would spend at the fast or slow station
    #TODO we need to pick actually reasonable values for delay_wgt so that it's on the same "scale" as price
            #after being multiplied by the wait time
        #right now it is drawn from one of two normal distributions (50% chance of being from each) with mean
            #values mean_wgt_ds, mean_wgt_ps. Then I bump it to 0 if it's negative
        #but finding the correct values of mean_wgt_ds, mean_wgt_ps is important
    
    ct=np.random.rand()
    if ct<0.5:
        delay_wgt=np.random.normal(mean_wgt_ds, mean_wgt_ps/2)
    else:
        delay_wgt=np.random.normal(mean_wgt_ps, mean_wgt_ps/2)
    if delay_wgt<0:
        delay_wgt=0

    #only randomly generate one, for fairness
    servetime_slow=np.random.exponential(mu_slow)
    servetime_fast=servetime_slow*(mu_fast/mu_slow)

    return delay_wgt,servetime_fast,servetime_slow

def arrival(L_slow,L_fast,mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps,c_fast,c_slow,p_fast,p_slow):
    #every time a customer arrives. Takes in the current state information, as well as constant params
    #Input: #L_slow,L_fast: the queue lengths (INCLUDING ppl charging) at time arr_time
            #mu_fast, mu_slow, mean_wgt_ds, mean_wgt_ps: constant params, defined in generate_cust()
            #c_fast,c_slow: num chargers of each type
            #p_fast,p_slow: price of each type. "constant" within each simulation, but we manually change between them
    #Output:L_slow,L_fast: new queue lengths after customer chooses a queue

    delay_wgt,servetime_fast,servetime_slow=generate_cust(mu_fast, mu_slow, mean_wgt_ds, mean_wgt_ps)

    charger_choice=choose_chargertype(delay_wgt,c_fast,c_slow,L_fast,L_slow,mu_fast,mu_slow,servetime_fast,servetime_slow,p_fast,p_slow)
    
    if charger_choice=="fast":
        L_fast=L_fast+1
    else:
        L_slow=L_slow+1
    
    #TODO: ALL this does rn is updates queue lengths. But also every person needs a total cost calculated. Use this method only if it's helpful, ignore otherwise
    return L_fast,L_slow

def simulate(n,p_fast,p_slow,mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps,c_fast,c_slow,lamb):
    #run the simulation
    #TODO: this is mostly pseudocode bc I don't remember how simulations work
    #Input: #n:number of customers to arrive
            #p_fast,p_slow: price PER MINUTE of fast and slow chargers, respectively. Change each simulation
            #mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps,c_fast,c_slow: constant params, defined where they are used
            #lamb: arrival rate into the system at large. Should pick based on the c and mu values to not blow up the system
    
    #TODO: generate n arrival times
    arrival_times=[1,2,3]

    #initial values for L_slow and L_fast (queue lengths). I have code to increase when someone arrivals but not to decrease when someone leaves
    L_slow=0
    L_fast=0

    for t in arrival_times:
    #TODO: I have no idea if this is the right structure for this. Also need some kind of array of service times? Idk. This is so so approximate pls feel free to change it
        L_fast,L_slow=arrival(L_slow,L_fast,mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps,c_fast,c_slow,p_fast,p_slow)
        #TODO calculate true service time for every cust
        #TODO calculate cost for every cust
    
    #TODO total_cost = sum of every customer's cost
    total_cost=0

    return total_cost

if __name__ == "__main__":
    #input params (do NOT change these between trials)
    #TODO change these obviously from what they are now
    n=0
    mu_fast=0.1
    mu_slow=0.1
    mean_wgt_ds=0
    mean_wgt_ps=0
    c_fast=0.1
    c_slow=0.1
    lamb=0.1

    #prices (change these between trials)
    p_fast=0.1
    p_slow=0.1

    total_cost=simulate(n,p_fast,p_slow,mu_fast,mu_slow,mean_wgt_ds,mean_wgt_ps,c_fast,c_slow,lamb)
    print(total_cost)