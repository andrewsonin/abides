import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd
from dateutil.parser import parse

from abides.agent import OrderBookImbalanceAgent
from abides.agent import exchange
from abides.agent.examples.MarketReplayAgentUSD import MarketReplayAgentUSD
from abides.agent.examples.MomentumAgent import MomentumAgent
from abides.order.base import LimitOrder
from abides.core import Kernel
from model.LatencyModel import LatencyModel
from abides import util

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for USD config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='10:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='11:30:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# Execution agent config
parser.add_argument('-e',
                    '--execution-agents',
                    action='store_true',
                    help='Flag to allow the execution agent to trade.')
parser.add_argument('-p',
                    '--execution-pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level for execution agent')
# market maker config
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-8,
                    help='Volatility of fundamental time series.'
                    )

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = False  # True
log_orders = None
book_freq = 0

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in RUB.

stream_history_length = 25000

agents.extend([exchange(agent_id=0,
                        name="EXCHANGE_AGENT",
                        type="ExchangeAgent",
                        mkt_open=mkt_open,
                        mkt_close=mkt_close,
                        symbols=[symbol],
                        log_orders=exchange_log_orders,
                        pipeline_delay=0,
                        computation_delay=0,
                        stream_history=stream_history_length,
                        book_freq=book_freq,
                        wide_book=True,
                        random_state=np.random.RandomState(
                            seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Market Replay Agent
file_name = 'LOB_df.pkl'
orders_file_path = f'./data/marketreplay/input/{file_name}'

agents.extend([MarketReplayAgentUSD(id=1,
                                    name="MARKET_REPLAY_AGENT",
                                    type='MarketReplayAgent',
                                    symbol=symbol,
                                    log_orders=False,
                                    date=historical_date,
                                    start_time=mkt_open,
                                    end_time=mkt_close,
                                    orders_file_path=orders_file_path,
                                    processed_orders_folder_path='./data/marketreplay/output/',
                                    starting_cash=0,
                                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                              dtype='uint64')))])
agent_types.extend("MarketReplayAgent")
agent_count += 1

# 5) Momentum Agents
num_momentum_agents = 1

agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1000,
                             max_size=10000,
                             wake_up_freq='20s',
                             log_orders=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")

# 5.1) imbalance agent
num_obi_agents = 1
agents.extend([OrderBookImbalanceAgent(agent_id=j, name="OBI_AGENT_{}".format(j), symbol=symbol, entry_threshold=0.2,
                                       freq=3600000000, starting_cash=starting_cash, log_orders=True,
                                       random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                 dtype='uint64')))
               for j in range(agent_count, agent_count + num_obi_agents)])
agent_types.extend("OrderBookImbalanceAgent")
agent_count += num_obi_agents

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("USD Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                        dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 0  # 50 nanoseconds #there was 50, doesn't work for MarketReplay as this is history that should be
# executed at exact order book time.


# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))
pairwise = (agent_count, agent_count)

# All agents sit on line from my PC to MICEX
me_to_micex_meters = 10000
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, me_to_micex_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args
                             )
# KERNEL

latency = np.zeros((agent_count, agent_count))  # TODO: check this way to setup separate latency for agent
noise = [0.0]

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              # agentLatency=latency,
              # latencyNoise=noise,
              oracle=None,
              log_dir=args.log_dir)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
