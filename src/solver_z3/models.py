try:
    import z3
except ImportError:
    z3 = None
from typing import List, Optional, Literal, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

class QuestionType(Enum):
    YES_NO           = "yes_no"
    MCQ_CORRECT      = "mcq_correct"
    FEWEST_PREMISES  = "fewest_premises"
    SCENARIO         = "scenario"
    PREMISE_TRACKING = "premise_tracking"
    OPEN_ENDED       = "open_ended"

# ---------------------------------------------------------------------------
# Proof granularity
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    step_number: int
    premise_index: int          # 0-based internally; converted to 1-based at output
    premise_fol: str
    produces: str               # intermediate or final conclusion (human-readable FOL)
    rule: str                   # inference rule name

@dataclass
class ProofPath:
    indices: List[int]          # 0-based premise indices (converted later)
    fol_strings: List[str]
    steps: Optional[List[ProofStep]] = None
    conclusion: str = ""
    proof_direction: str = "ENTAILED"   # "ENTAILED" | "REFUTED"
    
    # Research-grade metadata
    minimality_verified: bool = False   # True if confirmed minimal by MUS/MES
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.indices)

# ---------------------------------------------------------------------------
# Per-premise diagnostics
# ---------------------------------------------------------------------------

@dataclass
class PremiseResult:
    index: int                  # 0-based internally; converted to 1-based at output
    original_fol: str
    status: Literal["SAT", "UNSAT", "UNKNOWN"] = "UNKNOWN"
    role: Literal["CONTRIBUTING", "REDUNDANT", "CONFLICTING", "NEUTRAL"] = "NEUTRAL"
    is_core: bool = False
    conflicts_with: List[int] = field(default_factory=list)
    explanation: str = ""
    
    # Research-grade: is this premise individually satisfiable?
    individually_sat: Optional[bool] = None

# ---------------------------------------------------------------------------
# Question item (for internal multi-sentence processing)
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    label: str
    fol: str
    z3_expr: Any

# ---------------------------------------------------------------------------
# SolverTrace — captures Z3 call metadata for debugging / benchmarking
# ---------------------------------------------------------------------------

@dataclass
class SolverTrace:
    """Records each Z3 solver invocation for auditability."""
    call_id: int
    purpose: str                # e.g. "global_consistency", "entailment_check"
    result: str                 # "sat" | "unsat" | "unknown"
    elapsed_ms: float
    num_assertions: int
    reason_unknown: Optional[str] = None

# ---------------------------------------------------------------------------
# Top-level reasoning result
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    question_fol: str
    question_type: Optional[QuestionType] = None
    subtype: Optional[str] = None
    question_status: Literal[
        "ENTAILED", "REFUTED", "CONSISTENT", "PREMISE_CONFLICT", "UNKNOWN"
    ] = "UNKNOWN"
    premise_results: List[PremiseResult] = field(default_factory=list)
    proof_paths: Optional[List[ProofPath]] = None

    # FEWEST_PREMISES / MCQ
    mc_answer: Optional[str] = None
    mc_option_results: Optional[Dict] = None

    # SCENARIO
    witness_model: Optional[Dict[str, str]] = None

    # Conflict
    unsat_core: List[int] = field(default_factory=list)
    
    # Explanation (set at the end of solve())
    explanation: str = ""
    
    # Solver diagnostics
    z3_reason_unknown: Optional[str] = None
    solver_traces: List[SolverTrace] = field(default_factory=list)
    confidence: float = 1.0
    z3_time_ms: float = 0.0
    timeout_budget_ms: float = 0.0
    error: Optional[str] = None
    logic_draft: Optional[str] = None # Stores FOL draft for fallback hints
    z3_code: Optional[str] = None
    fol_premises: List[str] = field(default_factory=list)
    atomic_registry: Optional[Dict[str, str]] = None
    
    # Research metadata
    num_proof_paths: int = 0
    shortest_proof_size: Optional[int] = None

    def get_answer(self) -> str:
        if self.mc_answer:
            ans = str(self.mc_answer).strip()
            if ans.lower() in {"true", "yes", "entailed"}:
                return "True"
            if ans.lower() in {"false", "no", "refuted"}:
                return "False"
            if ans.lower() in {"unknown", "uncertain"}:
                return "Unknown"
            return ans
        
        if self.question_status == "ENTAILED":
            return "True"
        elif self.question_status == "REFUTED":
            return "False"
        
        return "Unknown"

    def finalize_proof_stats(self):
        """Call after proof_paths is fully populated."""
        if self.proof_paths:
            self.num_proof_paths = len(self.proof_paths)
            self.shortest_proof_size = min(p.size for p in self.proof_paths)
        else:
            self.num_proof_paths = 0
            self.shortest_proof_size = None

    def compute_confidence(self):
        """
        Heuristic confidence score:
          - Starts at 1.0
          - Penalised if solver returned unknown on any sub-call
          - Penalised when a single solver call nears its time budget (timeout pressure)
        """
        if self.question_status == "UNKNOWN":
            self.confidence = 0.0
            return
        if any(t.result == "unknown" for t in self.solver_traces):
            self.confidence = max(0.3, self.confidence - 0.4)
        # Soft penalty only when an *individual* solver call consumed >80% of its
        # allotted budget. Previously this compared the cumulative wall time against
        # the slowest single call, which was almost always >1.0 and so fired on every
        # multi-call solve — pinning confidence at 0.8 regardless of difficulty.
        if self.timeout_budget_ms > 0 and self.solver_traces:
            slowest_call_ms = max(t.elapsed_ms for t in self.solver_traces)
            if (slowest_call_ms / self.timeout_budget_ms) > 0.8:
                self.confidence = max(0.5, self.confidence - 0.2)
