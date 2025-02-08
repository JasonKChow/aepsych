from aepsych_client import AEPsychClient
from aepsych.transforms import ParameterTransforms
from aepsych.config import Config
import torch


def simulate_response(x, transform=None):
    # Unpack config
    values = list(x["config"].values())
    x = torch.tensor(values).T

    if transform is not None:
        x = transform.transform(x)

    # Normalize and simulate response
    x = x / x.shape[1]
    return float((torch.rand((1,)) < torch.sum(x)).item())


config_str = """
    [common]
    parnames = [par1, par2]
    stimuli_per_trial = 1
    outcome_types = [binary]
    strategy_names = [init_strat, opt_strat]

    [par1]
    par_type = continuous
    lower_bound = 1
    upper_bound = 10

    [par2]
    par_type = continuous
    lower_bound = 0
    upper_bound = 1

    [init_strat]
    generator = SobolGenerator
    min_asks = 10

    [opt_strat]
    model = GPClassificationModel
    generator = OptimizeAcqfGenerator
    min_asks = 50
    copy_model = True

    [OptimizeAcqfGenerator]
    acqf = MCLevelSetEstimation

    [EAVC]
    target = 0.75
"""

client = AEPsychClient(ip="0.0.0.0", port=5555)
transforms = ParameterTransforms.from_config(config=Config(config_str=config_str))

# Run experiment
n_trials = 0
while True:
    x = client.ask()
    print(f"Ask response {x}")
    y = simulate_response(x, transform=transforms)
    print(f"Simulated response: {y}")

    client.tell(config=x["config"], outcome=y)
    n_trials += 1
    print(f"Completed {n_trials} trials")
    if x["is_finished"]:
        break
