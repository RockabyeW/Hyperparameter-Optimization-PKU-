from smac.scenario.scenario import Scenario
def _get_optimizer_params(scoring_function, tunable_hyperparameters,
                          iterations, tmp_dir, **kwargs):
    config_space = _create_config_space(tunable_hyperparameters)
    tae_runner = _adapt_scoring_function(scoring_function)
    scenario = Scenario({
        'run_obj': 'quality',
        'runcount_limit': iterations,
        'cs': config_space,
        'deterministic': 'true',
        'output_dir': tmp_dir,
        'limit_resources': False,
    })

    optimizer_params = {
        'scenario': scenario,
        'rng': 42,
        'tae_runner': tae_runner,
    }

    if kwargs:
        optimizer_params.update(kwargs)

    return optimizer_params