"""
Signal definition and registry for Apollo.

A signal is a quantifiable variable hypothesized to predict football outcomes
better than (or differently from) market prices.

Every signal must:
1. Be computable WITHOUT knowing the match outcome
2. Have a clear causal mechanism
3. Be testable against bookmaker baseline
4. Survive multiple testing correction
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List
import yaml
from pathlib import Path


class SignalStatus(str, Enum):
    UNTESTED = "untested"
    TESTING = "testing"
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    INSUFFICIENT_DATA = "insufficient_data"
    REJECTED = "rejected"       # Failed robustness checks
    VALIDATED = "validated"     # Passed all checks including holdout


class SignalStrength(str, Enum):
    NONE = "none"
    WEAK = "weak"              # p < 0.05 but small effect
    MODERATE = "moderate"      # p < 0.01 and meaningful effect
    STRONG = "strong"          # p < 0.001 and large effect, cross-validated


@dataclass
class Signal:
    """Definition of a single forecasting signal."""
    
    signal_id: str              # e.g., "SIG_001"
    name: str                   # e.g., "Recent Form Bias"
    description: str            # What the signal measures
    mechanism: str              # Why it should work (causal theory)
    
    # Data requirements
    data_columns: List[str]     # Which columns it needs
    lookback_window: Optional[int] = None  # Matches of history needed
    
    # Results (filled after testing)
    status: SignalStatus = SignalStatus.UNTESTED
    strength: SignalStrength = SignalStrength.NONE
    
    brier_skill_score: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    roi: Optional[float] = None
    n_observations: Optional[int] = None
    
    # Cross-validation
    leagues_tested: List[str] = field(default_factory=list)
    leagues_significant: List[str] = field(default_factory=list)
    seasons_tested: int = 0
    seasons_significant: int = 0
    
    # Metadata
    experiment_id: Optional[str] = None
    notes: str = ""
    recommendation: str = ""    # "Use", "Monitor", "Reject"
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['strength'] = self.strength.value
        return d


class SignalRegistry:
    """
    Persistent catalog of all tested signals.
    Stored as YAML for human readability and version control.
    """
    
    def __init__(self, path: str = "signals/registry.yaml"):
        self.path = Path(path)
        self.signals: dict = {}  # signal_id -> Signal
        self._load()
    
    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                data = yaml.safe_load(f) or {}
            for sid, sdata in data.items():
                sdata['status'] = SignalStatus(sdata.get('status', 'untested'))
                sdata['strength'] = SignalStrength(sdata.get('strength', 'none'))
                self.signals[sid] = Signal(**sdata)
    
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: sig.to_dict() for sid, sig in self.signals.items()}
        with open(self.path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def register(self, signal: Signal):
        self.signals[signal.signal_id] = signal
        self.save()
    
    def update_results(self, signal_id: str, results: dict):
        if signal_id not in self.signals:
            raise KeyError(f"Signal {signal_id} not in registry")
        
        sig = self.signals[signal_id]
        for key, value in results.items():
            if hasattr(sig, key):
                setattr(sig, key, value)
        self.save()
    
    def summary(self) -> str:
        lines = ["Signal Registry Summary", "=" * 50]
        
        for status in SignalStatus:
            sigs = [s for s in self.signals.values() if s.status == status]
            if sigs:
                lines.append(f"\n{status.value.upper()} ({len(sigs)}):")
                for s in sigs:
                    bss = f"BSS={s.brier_skill_score:.4f}" if s.brier_skill_score else "—"
                    lines.append(f"  {s.signal_id}: {s.name} [{bss}]")
        
        return "\n".join(lines)
