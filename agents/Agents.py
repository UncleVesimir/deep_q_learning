from agents.deep_q_agent import DeepQAgent
from agents.double_deep_q_agent import DoubleDeepQAgent
from agents.dueling_double_deep_q_agent import DuelingDoubleDeepQAgent
from agents.distributional_q_agent import DistributionalDoubleDeepQAgent

agents_dict = {
    "DQN": DeepQAgent,
    "Double_DQN": DoubleDeepQAgent,
    "Dueling_Double_DQN": DuelingDoubleDeepQAgent,
    "Distributional_Double_DQN": DistributionalDoubleDeepQAgent
}

__all__ = ["DQN","Double_DQN", "Dueling_Double_DQN", "Distributional_Double_DQN", "agents_dict"]