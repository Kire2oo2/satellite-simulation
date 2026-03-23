import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim

def main():
  sim_config = {'t_0':0,'t_e':1,'t_step':1,'speed_factor':1,'anim_dt':1/25,'scale_factor':1,'visualise':True}
  scenario = sim.BaseScenario()
  sim.create_and_start_simulation(sim_config,scenario)

if __name__ == "__main__":
    main()
