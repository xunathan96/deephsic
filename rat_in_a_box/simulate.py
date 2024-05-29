import argparse
import dill
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells, VelocityCells, HeadDirectionCells

class RatInABox:

    def __init__(self, args):
        self.args = args
        self._build_env()
        self._build_cells()

    def _build_env(self):
        self.env = Environment(
            params={"scale": 1,
                    "boundary": [
                        [-0.5, -0.5],
                        [0.5, -0.5],
                        [0.5, 0.5],
                        [-0.5, 0.5]
                    ]})
        self.env.add_wall(np.array([[0, -0.5], [0, 0]]))
        # self.env.add_wall(np.array([[-0.25, -0.5], [-0.25, 0]]))
        # self.env.add_wall(np.array([[-0.5, 0.25], [0, 0.25]]))
        # self.env.add_wall(np.array([[0.25, 0.5], [0.25, 0]]))
        # self.env.add_wall(np.array([[0, -0.25], [0.5, -0.25]]))

        self.agent = Agent(
            self.env,
            params={"dt": self.args.dt})

    def _build_cells(self):
        self.PCs = PlaceCells(
            self.agent,
            params={"n": self.args.n_place_cells,
                    "description": "gaussian_threshold",
                    "widths": 0.2,
                    "wall_geometry": "geodesic",
                    "min_fr": self.args.min_fr,
                    "max_fr": self.args.max_fr,})

        self.GCs = GridCells(
            self.agent,
            params={"n": self.args.n_grid_cells,
                    "description": "three_rectified_cosines",  # can also be "three_shifted_cosines" as in Solstad 2006 Eq. (2)
                    "gridscale_distribution": "uniform",
                    "gridscale": (0.3, 0.7),
                    "min_fr": self.args.min_fr,
                    "max_fr": self.args.max_fr,})

        self.BVCs = BoundaryVectorCells(
            self.agent,
            params={"n": self.args.n_boundary_vector_cells,
                    "min_fr": self.args.min_fr,
                    "max_fr": self.args.max_fr,})

        self.HDCs = HeadDirectionCells(
            self.agent,
            params={"n": self.args.n_head_direction_cells,
                    "angular_spread_degrees": 35,
                    "min_fr": self.args.min_fr,
                    "max_fr": self.args.max_fr,}
        )

        self.VCs = VelocityCells(
            self.agent,
            params={"min_fr": self.args.min_fr,
                    "max_fr": self.args.max_fr,})

    def step(self):
        self.agent.update()
        self.PCs.update()
        self.GCs.update()
        self.BVCs.update()
        self.HDCs.update()
        self.VCs.update()

    def reset(self):
        self.agent.reset_history()
        self.PCs.reset_history()
        self.GCs.reset_history()
        self.BVCs.reset_history()
        self.HDCs.reset_history()
        self.VCs.reset_history()

    def simulate_simple(self):
        np.random.seed(self.args.seed)
        T = 10 * 60 #10 mins
        for i in tqdm(range(int(T / self.agent.dt))):
            self.step()


    def simulate(self,
                 n_traj: int = 1,
                 T: int = 10,
                 horizon: int = 20,):
        np.random.seed(self.args.seed)
        data = []
        T_start = 0
        for n in tqdm(range(n_traj)):
            # simulate trajectory
            for i in range(n_steps:=int(T / self.agent.dt)):
                self.step()
            # add traj as data sample
            traj = self.collate_traj(T_start,
                                     T_start+T,
                                     subsample = max(n_steps//horizon, 1),
                                     horizon = horizon)
            data.append(traj)
            self.reset()    # clear history buffer to release memory
            T_start += T
        return data


    def collate_traj(self,
                     t_start: int = None,
                     t_end: int = None,
                     subsample: int = 5,
                     horizon: int = 20,):
        r"""Collects all agent and neuron history for a trajectory in interval (t_start, t_end).
        The history is subsampled and only horizon number of steps are considered."""
        t = np.array(self.agent.history["t"])
        i_start = (0 if t_start is None else np.argmin(np.abs(t - t_start)))
        i_end = (-1 if t_end is None else np.argmin(np.abs(t - t_end)))           
        if (i_end - i_start + 1)//subsample < horizon:
            raise Warning("Trajectory length is smaller than the given horizon.")

        traj = dict()
        traj['agent'] = self.parse_agent_history(self.agent, i_start, i_end, subsample, horizon)
        traj['place'] = self.parse_neuron_history(self.PCs, i_start, i_end, subsample, horizon)
        traj['grid'] = self.parse_neuron_history(self.GCs, i_start, i_end, subsample, horizon)
        traj['boundary-vector'] = self.parse_neuron_history(self.BVCs, i_start, i_end, subsample, horizon)
        traj['head-direction'] = self.parse_neuron_history(self.HDCs, i_start, i_end, subsample, horizon)
        traj['velocity'] = self.parse_neuron_history(self.VCs, i_start, i_end, subsample, horizon)
        return traj


    @staticmethod
    def parse_agent_history(Agent: ratinabox.Agent,
                            i_start: int,
                            i_end: int,
                            subsample: int = 5,
                            horizon: int = 20,):
        traj = dict()
        for key, states in Agent.history.items():
            # subsample data for training (most of it is redundant anyway)
            traj[key] = np.array(states)[i_start:i_end:subsample][-horizon:]    # (T, D)
        # convert shape to (T, 1)
        traj['distance_travelled'] = traj['distance_travelled'][:, np.newaxis]
        traj['rot_vel'] = traj['rot_vel'][:, np.newaxis]
        return traj
        # Agent.history["t"]                    scalar
        # Agent.history["pos"]                  array [x,y]
        # Agent.history["distance_travelled"]   scalar
        # Agent.history["vel"]                  array [dx,dy]
        # Agent.history["rot_vel"]              scalar
        # Agent.history["head_direction"]       array [x,y]


    @staticmethod
    def parse_neuron_history(Neurons: ratinabox.Neurons,
                             i_start: int = None,
                             i_end: int = None,
                             subsample: int = 5,
                             horizon: int = 20,):
        traj = dict()
        for key, states in Neurons.history.items():
            traj[key] = np.array(states)[i_start:i_end:subsample][-horizon:]    # (T, D)
        return traj
        # Neurons.history["t"]              scalar
        # Neurons.history["firingrate"]     array [n0, ..., n15] of floats
        # Neurons.history["spikes"]         array [n0, ..., n15] of bools




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu',
                        action='store_true',
                        help='use cpu during experiement.')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='the gpu core to use during experiment.')
    parser.add_argument('--seed',
                        type=int,
                        default=0)
    parser.add_argument('--save-dir',
                        type=str,
                        default='./data',
                        help='directory to save data.')
    # simulation 
    parser.add_argument('--n-traj',
                        type=int,
                        default=10000,)
    parser.add_argument('--traj-duration',
                        type=int,
                        default=10,
                        help='duration of a single trajectory in seconds (default 10).')
    parser.add_argument('--traj-length',
                        type=int,
                        default=20,
                        help='number of steps for each trajectory (default 20).')
    # neuron
    parser.add_argument('--n-place-cells',
                        type=int,
                        default=16,)
    parser.add_argument('--n-grid-cells',
                        type=int,
                        default=16,)
    parser.add_argument('--n-boundary-vector-cells',
                        type=int,
                        default=16,)
    parser.add_argument('--n-head-direction-cells',
                        type=int,
                        default=16,)
    parser.add_argument('--receptive-field',
                        type=float,
                        default=0.2)
    parser.add_argument('--min-fr',
                        type=float,
                        default=0)
    parser.add_argument('--max-fr',
                        type=float,
                        default=1.)
    parser.add_argument('--dt',
                        type=float,
                        default=5e-2,)
    return parser.parse_args()




def test_simulation(args):
    riab = RatInABox(args)
    riab.simulate(n_traj = args.n_traj,
                  T = args.traj_duration,
                  horizon = args.traj_length)

    fig, ax = riab.env.plot_environment()
    riab.agent.plot_trajectory(fig=fig, ax=ax, alpha=0.5)
    plt.show()

    fig, ax = riab.PCs.plot_place_cell_locations()
    riab.PCs.plot_rate_map(chosen_neurons="all", method="groundtruth")
    riab.GCs.plot_rate_map(method='groundtruth', colorbar=False)
    riab.BVCs.plot_rate_map(method='groundtruth', colorbar=False)
    plt.show()



def main(args):
    riab = RatInABox(args)
    data = riab.simulate(n_traj = args.n_traj,
                         T = args.traj_duration,
                         horizon = args.traj_length)    
    # save data
    savefile = Path(args.save_dir)/'riab.pkl'
    savefile.parent.mkdir(parents=True, exist_ok=True)
    dump(data, savefile)
    # data = load(savefile)

    # visualize trajectories
    # fig, ax = riab.env.plot_environment()
    # riab.agent.plot_trajectory(fig=fig, ax=ax, alpha=0.5)
    # plt.show()

    fig, ax = riab.env.plot_environment()
    for traj in data:
        pos = traj['agent']['pos']
        plt.plot(pos[:,0], pos[:,1])
    plt.show()


    # (GCs, BVCs) vs (pos, vel, rot_vel, head_direction) for t+1, t+2 to make it high dim
    # seq-to-seq deep kernel
    # X is a seq and Y is a seq



# =============================
#       HELPER FUNCTIONS
# =============================

def dump(obj, file):
    with open(file, 'wb') as f:
        dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)

def load(file):
    with open(file, 'rb') as f:
        obj = dill.load(f)
    return obj




if __name__=='__main__':
    main(parse_args())


