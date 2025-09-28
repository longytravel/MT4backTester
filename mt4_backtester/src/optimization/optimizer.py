"""Optimization framework with parallel processing support.

This module provides high-performance parameter optimization using
multiple optimization algorithms and parallel processing capabilities.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from ..core.engine import Backtester, EngineConfig
from ..core.result import BacktestResult
from ..core.strategy import Strategy


@dataclass
class OptimizationResult:
    """Result from optimization run."""

    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]]
    optimization_time: float
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    optimization_method: str
    total_trials: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'optimization_method': self.optimization_method,
            'total_trials': self.total_trials,
            'optimization_time': self.optimization_time,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save optimization results to file.

        Args:
            path: File path for saving results
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save detailed trial data
        trials_df = pd.DataFrame(self.all_trials)
        trials_path = path.with_suffix('.csv')
        trials_df.to_csv(trials_path, index=False)

        logger.info(f"Saved optimization results to {path}")


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    method: str = "optuna"  # optuna, grid, random, genetic
    n_trials: int = 100
    n_jobs: int = -1  # -1 for all CPU cores
    objective: str = "sharpe"  # sharpe, profit, win_rate, profit_factor
    direction: str = "maximize"  # maximize or minimize
    timeout: Optional[float] = None  # Timeout in seconds
    show_progress: bool = True
    save_results: bool = True
    results_path: str = "./optimization_results"
    seed: Optional[int] = 42

    # Optuna specific
    optuna_sampler: str = "TPE"  # TPE, CMA-ES, Random
    optuna_pruner: Optional[str] = "MedianPruner"  # MedianPruner, HyperbandPruner

    # Grid search specific
    grid_exhaustive: bool = True  # Test all combinations

    # Genetic algorithm specific
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8


class Optimizer:
    """Universal optimizer for trading strategies.

    Supports multiple optimization algorithms and parallel processing
    for high-performance parameter search.
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        engine_config: Optional[EngineConfig] = None
    ):
        """Initialize optimizer.

        Args:
            config: Optimization configuration
            engine_config: Backtesting engine configuration
        """
        self.config = config or OptimizationConfig()
        self.engine_config = engine_config or EngineConfig()

        # Set number of parallel jobs
        if self.config.n_jobs == -1:
            self.config.n_jobs = mp.cpu_count()

        logger.info(
            f"Initialized optimizer with {self.config.n_jobs} parallel jobs"
        )

    def optimize(
        self,
        strategy_class: type[Strategy],
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        param_space: Optional[Dict[str, Any]] = None,
        data_source: str = "auto",
        timeframe: Optional[str] = None,
    ) -> OptimizationResult:
        """Run optimization for a strategy.

        Args:
            strategy_class: Strategy class to optimize
            symbol: Trading symbol
            start: Start date
            end: End date
            param_space: Parameter search space (uses strategy default if None)
            data_source: Data source for backtesting
            timeframe: Primary timeframe

        Returns:
            OptimizationResult with best parameters and performance
        """
        logger.info(
            f"Starting {self.config.method} optimization for {strategy_class.__name__}"
        )

        # Get parameter space from strategy if not provided
        if param_space is None:
            dummy_strategy = strategy_class()
            param_space = dummy_strategy.get_optimization_params()

        if not param_space:
            raise ValueError("No parameters to optimize")

        # Convert dates
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        # Start optimization
        start_time = time.time()

        if self.config.method == "optuna":
            result = self._optimize_optuna(
                strategy_class, symbol, start, end,
                param_space, data_source, timeframe
            )
        elif self.config.method == "grid":
            result = self._optimize_grid(
                strategy_class, symbol, start, end,
                param_space, data_source, timeframe
            )
        elif self.config.method == "random":
            result = self._optimize_random(
                strategy_class, symbol, start, end,
                param_space, data_source, timeframe
            )
        elif self.config.method == "genetic":
            result = self._optimize_genetic(
                strategy_class, symbol, start, end,
                param_space, data_source, timeframe
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")

        # Finalize result
        result.optimization_time = time.time() - start_time
        result.strategy_name = strategy_class.__name__
        result.symbol = symbol
        result.start_date = start
        result.end_date = end
        result.optimization_method = self.config.method

        # Save results if configured
        if self.config.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{strategy_class.__name__}_{symbol}_{timestamp}.json"
            path = Path(self.config.results_path) / filename
            result.save(path)

        logger.info(
            f"Optimization completed in {result.optimization_time:.2f}s. "
            f"Best score: {result.best_score:.4f}"
        )

        return result

    def _optimize_optuna(
        self,
        strategy_class: type[Strategy],
        symbol: str,
        start: datetime,
        end: datetime,
        param_space: Dict[str, Any],
        data_source: str,
        timeframe: Optional[str]
    ) -> OptimizationResult:
        """Run Optuna optimization.

        Args:
            strategy_class: Strategy class
            symbol: Trading symbol
            start: Start date
            end: End date
            param_space: Parameter space
            data_source: Data source
            timeframe: Timeframe

        Returns:
            Optimization result
        """

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            # Sample parameters
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range
                    )
                elif isinstance(param_range, range):
                    # Integer parameter
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_range.start,
                        param_range.stop - param_range.step,
                        param_range.step
                    )
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    # Float parameter (min, max)
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1]
                    )
                else:
                    # Fixed value
                    params[param_name] = param_range

            # Run backtest
            score = self._evaluate_params(
                strategy_class, params, symbol, start, end,
                data_source, timeframe
            )

            return score

        # Create Optuna study
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=self._get_optuna_sampler(),
            pruner=self._get_optuna_pruner(),
        )

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            show_progress_bar=self.config.show_progress,
        )

        # Extract results
        all_trials = [
            {**trial.params, 'score': trial.value, 'state': str(trial.state)}
            for trial in study.trials
        ]

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_trials=all_trials,
            optimization_time=0,  # Will be set by caller
            strategy_name="",  # Will be set by caller
            symbol=symbol,
            start_date=start,
            end_date=end,
            optimization_method=self.config.method,
            total_trials=len(study.trials),
        )

    def _optimize_grid(
        self,
        strategy_class: type[Strategy],
        symbol: str,
        start: datetime,
        end: datetime,
        param_space: Dict[str, Any],
        data_source: str,
        timeframe: Optional[str]
    ) -> OptimizationResult:
        """Run grid search optimization.

        Args:
            strategy_class: Strategy class
            symbol: Trading symbol
            start: Start date
            end: End date
            param_space: Parameter space
            data_source: Data source
            timeframe: Timeframe

        Returns:
            Optimization result
        """
        # Generate all parameter combinations
        param_combinations = self._generate_grid_combinations(param_space)

        logger.info(f"Grid search: testing {len(param_combinations)} combinations")

        # Evaluate all combinations in parallel
        if self.config.show_progress:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._evaluate_params)(
                    strategy_class, params, symbol, start, end,
                    data_source, timeframe
                )
                for params in tqdm(param_combinations, desc="Grid Search")
            )
        else:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._evaluate_params)(
                    strategy_class, params, symbol, start, end,
                    data_source, timeframe
                )
                for params in param_combinations
            )

        # Find best result
        all_trials = [
            {**params, 'score': score}
            for params, score in zip(param_combinations, results)
        ]

        best_idx = np.argmax(results) if self.config.direction == "maximize" else np.argmin(results)
        best_params = param_combinations[best_idx]
        best_score = results[best_idx]

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            optimization_time=0,
            strategy_name="",
            symbol=symbol,
            start_date=start,
            end_date=end,
            optimization_method=self.config.method,
            total_trials=len(param_combinations),
        )

    def _optimize_random(
        self,
        strategy_class: type[Strategy],
        symbol: str,
        start: datetime,
        end: datetime,
        param_space: Dict[str, Any],
        data_source: str,
        timeframe: Optional[str]
    ) -> OptimizationResult:
        """Run random search optimization.

        Args:
            strategy_class: Strategy class
            symbol: Trading symbol
            start: Start date
            end: End date
            param_space: Parameter space
            data_source: Data source
            timeframe: Timeframe

        Returns:
            Optimization result
        """
        np.random.seed(self.config.seed)

        # Generate random parameter combinations
        param_combinations = []
        for _ in range(self.config.n_trials):
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list):
                    params[param_name] = np.random.choice(param_range)
                elif isinstance(param_range, range):
                    params[param_name] = np.random.choice(list(param_range))
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    params[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    params[param_name] = param_range
            param_combinations.append(params)

        # Evaluate combinations in parallel
        if self.config.show_progress:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._evaluate_params)(
                    strategy_class, params, symbol, start, end,
                    data_source, timeframe
                )
                for params in tqdm(param_combinations, desc="Random Search")
            )
        else:
            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._evaluate_params)(
                    strategy_class, params, symbol, start, end,
                    data_source, timeframe
                )
                for params in param_combinations
            )

        # Find best result
        all_trials = [
            {**params, 'score': score}
            for params, score in zip(param_combinations, results)
        ]

        best_idx = np.argmax(results) if self.config.direction == "maximize" else np.argmin(results)
        best_params = param_combinations[best_idx]
        best_score = results[best_idx]

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            optimization_time=0,
            strategy_name="",
            symbol=symbol,
            start_date=start,
            end_date=end,
            optimization_method=self.config.method,
            total_trials=self.config.n_trials,
        )

    def _optimize_genetic(
        self,
        strategy_class: type[Strategy],
        symbol: str,
        start: datetime,
        end: datetime,
        param_space: Dict[str, Any],
        data_source: str,
        timeframe: Optional[str]
    ) -> OptimizationResult:
        """Run genetic algorithm optimization.

        Simplified implementation for demonstration.

        Args:
            strategy_class: Strategy class
            symbol: Trading symbol
            start: Start date
            end: End date
            param_space: Parameter space
            data_source: Data source
            timeframe: Timeframe

        Returns:
            Optimization result
        """
        logger.info(
            f"Genetic algorithm: {self.config.population_size} population, "
            f"{self.config.generations} generations"
        )

        # Initialize population
        population = self._initialize_population(
            param_space, self.config.population_size
        )

        all_trials = []
        best_params = None
        best_score = float('-inf') if self.config.direction == "maximize" else float('inf')

        for generation in range(self.config.generations):
            # Evaluate fitness
            fitness_scores = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._evaluate_params)(
                    strategy_class, individual, symbol, start, end,
                    data_source, timeframe
                )
                for individual in population
            )

            # Track all trials
            for params, score in zip(population, fitness_scores):
                all_trials.append({**params, 'score': score, 'generation': generation})

            # Update best
            if self.config.direction == "maximize":
                gen_best_idx = np.argmax(fitness_scores)
                if fitness_scores[gen_best_idx] > best_score:
                    best_score = fitness_scores[gen_best_idx]
                    best_params = population[gen_best_idx].copy()
            else:
                gen_best_idx = np.argmin(fitness_scores)
                if fitness_scores[gen_best_idx] < best_score:
                    best_score = fitness_scores[gen_best_idx]
                    best_params = population[gen_best_idx].copy()

            logger.info(
                f"Generation {generation + 1}/{self.config.generations}: "
                f"Best score = {best_score:.4f}"
            )

            # Select, crossover, and mutate for next generation
            if generation < self.config.generations - 1:
                population = self._evolve_population(
                    population, fitness_scores, param_space
                )

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            optimization_time=0,
            strategy_name="",
            symbol=symbol,
            start_date=start,
            end_date=end,
            optimization_method=self.config.method,
            total_trials=len(all_trials),
        )

    def _evaluate_params(
        self,
        strategy_class: type[Strategy],
        params: Dict[str, Any],
        symbol: str,
        start: datetime,
        end: datetime,
        data_source: str,
        timeframe: Optional[str]
    ) -> float:
        """Evaluate a parameter set.

        Args:
            strategy_class: Strategy class
            params: Parameters to evaluate
            symbol: Trading symbol
            start: Start date
            end: End date
            data_source: Data source
            timeframe: Timeframe

        Returns:
            Objective score
        """
        try:
            # Create strategy with parameters
            strategy = strategy_class(**params)

            # Run backtest
            backtester = Backtester(config=self.engine_config)
            result = backtester.run(
                strategy=strategy,
                symbol=symbol,
                start=start,
                end=end,
                data_source=data_source,
                timeframe=timeframe,
            )

            # Calculate objective score
            if self.config.objective == "sharpe":
                return result.sharpe_ratio
            elif self.config.objective == "profit":
                return result.net_profit
            elif self.config.objective == "win_rate":
                return result.win_rate
            elif self.config.objective == "profit_factor":
                return result.profit_factor
            else:
                raise ValueError(f"Unknown objective: {self.config.objective}")

        except Exception as e:
            logger.warning(f"Error evaluating params {params}: {e}")
            return float('-inf') if self.config.direction == "maximize" else float('inf')

    def _generate_grid_combinations(
        self, param_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search.

        Args:
            param_space: Parameter space

        Returns:
            List of parameter combinations
        """
        from itertools import product

        # Convert param_space to lists
        param_names = []
        param_values = []

        for name, values in param_space.items():
            param_names.append(name)
            if isinstance(values, (list, range)):
                param_values.append(list(values))
            else:
                param_values.append([values])

        # Generate all combinations
        combinations = []
        for values in product(*param_values):
            combinations.append(dict(zip(param_names, values)))

        return combinations

    def _get_optuna_sampler(self):
        """Get Optuna sampler based on configuration."""
        if self.config.optuna_sampler == "TPE":
            return optuna.samplers.TPESampler(seed=self.config.seed)
        elif self.config.optuna_sampler == "CMA-ES":
            return optuna.samplers.CmaEsSampler(seed=self.config.seed)
        elif self.config.optuna_sampler == "Random":
            return optuna.samplers.RandomSampler(seed=self.config.seed)
        else:
            return optuna.samplers.TPESampler(seed=self.config.seed)

    def _get_optuna_pruner(self):
        """Get Optuna pruner based on configuration."""
        if self.config.optuna_pruner == "MedianPruner":
            return optuna.pruners.MedianPruner()
        elif self.config.optuna_pruner == "HyperbandPruner":
            return optuna.pruners.HyperbandPruner()
        else:
            return None

    def _initialize_population(
        self, param_space: Dict[str, Any], size: int
    ) -> List[Dict[str, Any]]:
        """Initialize population for genetic algorithm.

        Args:
            param_space: Parameter space
            size: Population size

        Returns:
            Initial population
        """
        population = []
        for _ in range(size):
            individual = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, list):
                    individual[param_name] = np.random.choice(param_range)
                elif isinstance(param_range, range):
                    individual[param_name] = np.random.choice(list(param_range))
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    individual[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    individual[param_name] = param_range
            population.append(individual)
        return population

    def _evolve_population(
        self,
        population: List[Dict[str, Any]],
        fitness_scores: List[float],
        param_space: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evolve population using selection, crossover, and mutation.

        Args:
            population: Current population
            fitness_scores: Fitness scores
            param_space: Parameter space

        Returns:
            New population
        """
        new_population = []
        pop_size = len(population)

        # Convert scores for selection (higher is better)
        if self.config.direction == "minimize":
            fitness_scores = [-s for s in fitness_scores]

        # Normalize fitness scores for selection probabilities
        fitness_array = np.array(fitness_scores)
        fitness_array = fitness_array - fitness_array.min() + 1e-10
        probabilities = fitness_array / fitness_array.sum()

        # Elitism: keep best individual
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx].copy())

        # Generate rest of population
        while len(new_population) < pop_size:
            # Selection
            parent1_idx = np.random.choice(pop_size, p=probabilities)
            parent2_idx = np.random.choice(pop_size, p=probabilities)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child = self._mutate(child, param_space)

            new_population.append(child)

        return new_population[:pop_size]

    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Child individual
        """
        child = {}
        for key in parent1:
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(
        self, individual: Dict[str, Any], param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mutate an individual.

        Args:
            individual: Individual to mutate
            param_space: Parameter space

        Returns:
            Mutated individual
        """
        mutated = individual.copy()

        # Mutate one random parameter
        param_to_mutate = np.random.choice(list(param_space.keys()))
        param_range = param_space[param_to_mutate]

        if isinstance(param_range, list):
            mutated[param_to_mutate] = np.random.choice(param_range)
        elif isinstance(param_range, range):
            mutated[param_to_mutate] = np.random.choice(list(param_range))
        elif isinstance(param_range, tuple) and len(param_range) == 2:
            mutated[param_to_mutate] = np.random.uniform(param_range[0], param_range[1])

        return mutated


def export_to_mt4_set(
    params: Dict[str, Any],
    output_path: Union[str, Path],
    ea_name: str = "FairPrice"
) -> None:
    """Export optimized parameters to MT4 .set file.

    Args:
        params: Optimized parameters
        output_path: Path for .set file
        ea_name: EA name for the set file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map Python parameters to MT4 parameters
    mt4_params = {
        'MA_Period': params.get('ma_period', 200),
        'Slow_MA_Period': params.get('filter_period', 800),
        'Use_Slow_MA_Filter': params.get('use_filter', True),
        'Initial_Trigger_Pips': params.get('initial_trigger_pips', 100),
        'NumberOfPendingOrders': params.get('pending_order_count', 10),
        'PendingOrderRangePips': params.get('pending_order_range_pips', 50),
        'LotSize': params.get('lot_size', 0.01),
        'MagicNumber': params.get('magic_number', 12345),
        'Equity_StopOut_Percent': params.get('equity_stop_percent', 5.0),
        'Close_At_MA': params.get('close_at_ma', True),
    }

    # Write .set file
    with open(output_path, 'w') as f:
        f.write(f"; {ea_name} Expert Advisor Settings\n")
        f.write(f"; Generated: {datetime.now()}\n")
        f.write(f"; Optimized for maximum Sharpe ratio\n\n")

        for param_name, param_value in mt4_params.items():
            # Convert boolean to MT4 format
            if isinstance(param_value, bool):
                param_value = 'true' if param_value else 'false'

            f.write(f"{param_name}={param_value}\n")

    logger.info(f"Exported MT4 settings to {output_path}")