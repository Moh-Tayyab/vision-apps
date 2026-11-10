"""
Phase 1 Model & Dataset Architecture Package.

Provides:
  - Canonical two-class contract (0: carton, 1: person)
  - Person occluder mask and camera visibility scoring
  - 25-session manifest with zero-leakage split enforcement
  - Event-level ground truth CSV management and temporal bipartite matching
  - Multi-tier inference engine management with GPU/CPU fallback
  - Live shadow-mode execution and audit trail logging
  - Exact Clopper-Pearson 95% statistical certification
  - Automated dataset integrity and annotation QA
"""

from .class_contract import (
    CLASS_CARTON,
    CLASS_PERSON,
    CLASS_NAMES,
    CLASS_IDS,
    DetectionBox,
    InvalidClassContractError,
    is_carton,
    is_person,
    validate_carton_event_candidate,
    partition_detections,
)

from .person_occlusion import (
    CorridorZone,
    PersonOccluderMask,
    OcclusionResult,
)

from .session_manifest import (
    SessionRecord,
    DatasetManifest,
    ManifestValidationError,
)

from .event_ground_truth import (
    GroundTruthEvent,
    PredictedEvent,
    EventEvaluationMetrics,
    EventMatcher,
)

from .engine_manager import (
    InferenceEngineManager,
    EngineInitializationError,
    EngineBenchmarkResult,
)

from .shadow_mode import (
    ShadowModeAuditor,
    ShadowEventRecord,
)

from .statistical_certifier import (
    StatisticalCertifier,
    ClopperPearsonResult,
    compute_clopper_pearson_lower_bound,
)

from .dataset_validator import (
    DatasetValidator,
    DatasetQAReport,
)

__all__ = [
    "CLASS_CARTON",
    "CLASS_PERSON",
    "CLASS_NAMES",
    "CLASS_IDS",
    "DetectionBox",
    "InvalidClassContractError",
    "is_carton",
    "is_person",
    "validate_carton_event_candidate",
    "partition_detections",
    "CorridorZone",
    "PersonOccluderMask",
    "OcclusionResult",
    "SessionRecord",
    "DatasetManifest",
    "ManifestValidationError",
    "GroundTruthEvent",
    "PredictedEvent",
    "EventEvaluationMetrics",
    "EventMatcher",
    "InferenceEngineManager",
    "EngineInitializationError",
    "EngineBenchmarkResult",
    "ShadowModeAuditor",
    "ShadowEventRecord",
    "StatisticalCertifier",
    "ClopperPearsonResult",
    "compute_clopper_pearson_lower_bound",
    "DatasetValidator",
    "DatasetQAReport",
]
