from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.gcivl_vib import GCIVLVIBAgent
from agents.gcivl_cl import GCIVLCLAgent
from agents.gcivl_hilp import GCIVLHILPAgent
from agents.crl_hilp import CRLHILPAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    gcivl_vib=GCIVLVIBAgent,
    gcivl_cl=GCIVLCLAgent,
    gcivl_hilp=GCIVLHILPAgent,
    crl_hilp=CRLHILPAgent,
)
