from nika.net_env.containerlab.base import ContainerlabNetworkEnv


class ContainerlabSrlCeos01(ContainerlabNetworkEnv):
    LAB_NAME = "srlceos01"
    TOPO_LEVEL = "easy"
    TOPO_SIZE = None
    TAGS = ["link", "srl", "ceos", "containerlab"]
    DESC = "Nokia SR Linux and Arista cEOS interconnect (Containerlab srlceos01)."
