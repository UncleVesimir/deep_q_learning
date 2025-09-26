from agents.deep_q_agent import DeepQAgent
from agents.double_deep_q_agent import DoubleDeepQAgent
from agents.dueling_double_deep_q_agent import DuelingDoubleDeepQAgent

agents_dict = {
    "DQN": DeepQAgent,
    "Double_DQN": DoubleDeepQAgent,
    "Dueling_Double_DQN": DuelingDoubleDeepQAgent 
}

__all__ = ["DQN","Double_DQN", "Dueling_Double_DQN"]