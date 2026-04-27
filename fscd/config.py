from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_ALGORITHMS = ("pc", "ges", "boss")
DEFAULT_NODES = (5, 10, 15)
DEFAULT_DENSITIES = (0.2, 0.5, 0.8)
DEFAULT_SAMPLE_SIZES = (20, 50, 100, 200, 300, 1000, 5000, 10000)
DEFAULT_RUNS = 100
DEFAULT_SEED = 0
DEFAULT_OUTPUT = "results/pdf_default"
DEFAULT_CHECKPOINT_INTERVAL = 25


@dataclass
class BenchmarkConfig:
    algorithms: tuple[str, ...]
    nodes: tuple[int, ...]
    densities: tuple[float, ...]
    sample_sizes: tuple[int, ...]
    runs: int
    seed: int
    output_dir: Path
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL

    @classmethod
    def default(cls, output_dir: str | Path = DEFAULT_OUTPUT) -> "BenchmarkConfig":
        return cls(
            algorithms=DEFAULT_ALGORITHMS,
            nodes=DEFAULT_NODES,
            densities=DEFAULT_DENSITIES,
            sample_sizes=DEFAULT_SAMPLE_SIZES,
            runs=DEFAULT_RUNS,
            seed=DEFAULT_SEED,
            output_dir=Path(output_dir),
            checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
        )

    @classmethod
    def from_namespace(cls, namespace: object) -> "BenchmarkConfig":
        config = cls(
            algorithms=tuple(str(name).lower() for name in namespace.algorithms),
            nodes=tuple(int(value) for value in namespace.nodes),
            densities=tuple(float(value) for value in namespace.densities),
            sample_sizes=tuple(int(value) for value in namespace.sample_sizes),
            runs=int(namespace.runs),
            seed=int(namespace.seed),
            output_dir=Path(namespace.output),
            checkpoint_interval=int(namespace.checkpoint_interval),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.algorithms:
            raise ValueError("At least one algorithm must be specified.")
        if not self.nodes or any(node <= 0 for node in self.nodes):
            raise ValueError("All node counts must be positive.")
        if not self.densities or any(density < 0.0 or density > 1.0 for density in self.densities):
            raise ValueError("All densities must be in [0, 1].")
        if not self.sample_sizes or any(size <= 0 for size in self.sample_sizes):
            raise ValueError("All sample sizes must be positive.")
        if self.runs <= 0:
            raise ValueError("Runs must be positive.")
        if self.checkpoint_interval <= 0:
            raise ValueError("Checkpoint interval must be positive.")

    def to_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata["output_dir"] = str(self.output_dir)
        return metadata

