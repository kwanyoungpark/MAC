# Baselines (Model-free)
from agents.gciql import GCIQLAgent
from agents.ngcsacbc import NGCSACBCAgent
from agents.sharsa import SHARSAAgent

# Baselines (Model-based)
from agents.fmpc import FMPCAgent
from agents.leq import LEQAgent
from agents.mopo import MOPOAgent
from agents.mobile import MOBILEAgent

# Our method
from agents.mac import MACAgent

# Analysis
from agents.mbrs_ac import ACMBRSAgent   # MAC (Gau)
from agents.mbfql import MBFQLAgent      # MAC (FQL)
from agents.model_ac import ACModelAgent # Model inaccuracy analysis

agents = dict(
    gciql=GCIQLAgent,
    ngcsacbc=NGCSACBCAgent,
    sharsa=SHARSAAgent,

    fmpc=FMPCAgent,
    leq=LEQAgent,
    mopo=MOPOAgent,
    mobile=MOBILEAgent,

    mac=MACAgent,

    mbrs_ac=ACMBRSAgent,
    mbfql=MBFQLAgent,
    model_ac=ACModelAgent,
)
