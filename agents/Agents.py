from agents.deep_q_agent import DeepQAgent
from agents.double_deep_q_agent import DoubleDeepQAgent

agents_dict = {
    "DQN": DeepQAgent,
    "Double_DQN": DoubleDeepQAgent
}

__all__ = ["DQN","Double_DQN"]