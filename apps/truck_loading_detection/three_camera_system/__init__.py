"""
Three-Camera Carton Counting System Package (Self-Contained).

Consolidates all 3-camera detection, tracking, corridor analysis,
synchronization, ring buffering, and consensus logic in one dedicated package.
"""

from .consensus_policy import (
    ConsensusLevel,
    ConsensusVerdict,
    DecisionAction,
    ThreeCameraConsensusPolicy,
)
from .corridor import (
    CameraTripwireConfig,
    CorridorZone,
    CrossingCorridor,
    LocalCameraCrossingCandidate,
    LocalCrossingCandidate,
    PerCameraCorridorEvidence,
    PerCameraTripwire,
)
from .cross_camera_associator import (
    AssociatedCartonCluster,
    CrossCameraAssociator,
)
from .crossing_state_machine import (
    CartonCrossingEntity,
    CartonCrossingStateMachine,
    CommittedCrossingEvent,
    CrossingState,
)
from .detector_adapter import (
    CanonicalCartonDetection,
    CartonDetectorAdapter,
)
from .geometric_adapter import (
    CameraCalibrationProfile,
    GeometricCalibrationManager,
)
from .per_camera_tracker import (
    CartonTracklet,
    PerCameraCartonTracker,
    TrackStatus,
)
from .pipeline import (
    FrameTickOutput,
    ThreeCameraCartonPipeline,
)
from .ring_buffer import (
    BufferedFrame,
    PendingClipTrigger,
    SynchronizedRingBuffer,
)
from .stack_disambiguator import (
    StackDisambiguationResult,
    StackDisambiguator,
    StackResolutionState,
)
from .evaluator import (
    CartonLevelEvaluator,
    EventMatchResult,
    PalletScaleAuditResult,
    PalletScaleCrossChecker,
    PalletScaleReading,
    Week5EvaluationReport,
)
from .timeline_aligner import (
    CameraSyncConfig,
    SynchronizedTimestamp,
    TimelineAligner,
)
from .event_store import EventStore
from .shadow_mode import ShadowModeManager, ShadowStreamHealth, ShadowIncident
from .daemon import DaemonConfig, TruckLoadingProductionDaemon

__all__ = [
    "CanonicalCartonDetection",
    "CartonDetectorAdapter",
    "CartonTracklet",
    "PerCameraCartonTracker",
    "TrackStatus",
    "SynchronizedTimestamp",
    "TimelineAligner",
    "CameraSyncConfig",
    "CameraCalibrationProfile",
    "GeometricCalibrationManager",
    "AssociatedCartonCluster",
    "CrossCameraAssociator",
    "CartonCrossingStateMachine",
    "CommittedCrossingEvent",
    "CrossingState",
    "CartonCrossingEntity",
    "ThreeCameraConsensusPolicy",
    "ConsensusVerdict",
    "ConsensusLevel",
    "DecisionAction",
    "StackDisambiguator",
    "StackResolutionState",
    "StackDisambiguationResult",
    "ThreeCameraCartonPipeline",
    "FrameTickOutput",
    "CameraTripwireConfig",
    "LocalCameraCrossingCandidate",
    "LocalCrossingCandidate",
    "PerCameraTripwire",
    "CorridorZone",
    "CrossingCorridor",
    "PerCameraCorridorEvidence",
    "BufferedFrame",
    "PendingClipTrigger",
    "SynchronizedRingBuffer",
    "CartonLevelEvaluator",
    "PalletScaleCrossChecker",
    "PalletScaleReading",
    "PalletScaleAuditResult",
    "Week5EvaluationReport",
    "EventMatchResult",
    "EventStore",
    "ShadowModeManager",
    "ShadowStreamHealth",
    "ShadowIncident",
    "DaemonConfig",
    "TruckLoadingProductionDaemon",
]

