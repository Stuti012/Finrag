"""Configuration for the Financial QA System."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """LLM configuration."""

    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    load_in_4bit: bool = True
    device_map: str = "auto"


@dataclass
class RetrievalConfig:
    """Retrieval system configuration."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k_tables: int = 3
    top_k_text: int = 5
    chunk_size: int = 256
    chunk_overlap: int = 50
    use_bm25: bool = True
    bm25_weight: float = 0.3
    dense_weight: float = 0.7


@dataclass
class NumericalReasoningConfig:
    """Numerical reasoning module configuration."""

    max_program_steps: int = 10
    execution_timeout: int = 5
    verify_results: bool = True
    tolerance: float = 1e-4


@dataclass
class TemporalReasoningConfig:
    """Temporal reasoning module configuration."""

    max_hops: int = 3
    temporal_embedding_dim: int = 128
    use_graph: bool = True
    enable_temporal_causal_integration: bool = True
    temporal_granularity: str = "mixed"  # year|quarter|month|mixed
    trend_confidence_threshold: float = 0.4


@dataclass
class CausalityConfig:
    """Causality detection module configuration."""

    causal_embedding_dim: int = 128
    num_attention_heads: int = 4
    max_causal_hops: int = 3
    confidence_threshold: float = 0.5
    chain_min_confidence: float = 0.2
    enable_counterfactuals: bool = True
    enable_strength_estimation: bool = True


@dataclass
class TemporalCausalJointConfig:
    """Joint temporal-causal reasoning configuration."""

    enabled: bool = True
    joint_detection_threshold: float = 0.35
    attention_temperature: float = 1.0
    min_alignment_score: float = 0.3
    enable_cross_module_attention: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    numerical_tolerance: float = 0.01
    program_accuracy_weight: float = 0.4
    execution_accuracy_weight: float = 0.6
    causal_chain_weight: float = 0.2
    temporal_causal_alignment_weight: float = 0.2


@dataclass
class FinQAConfig:
    """Master configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    numerical: NumericalReasoningConfig = field(default_factory=NumericalReasoningConfig)
    temporal: TemporalReasoningConfig = field(default_factory=TemporalReasoningConfig)
    causality: CausalityConfig = field(default_factory=CausalityConfig)
    temporal_causal_joint: TemporalCausalJointConfig = field(default_factory=TemporalCausalJointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dataset_dir: str = "./finqa_data"
    output_dir: str = "./outputs"
    seed: int = 42
