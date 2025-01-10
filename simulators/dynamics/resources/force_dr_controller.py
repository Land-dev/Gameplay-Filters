import numpy as np

class NaiveForceDRController():
    def __init__(self, force_info=None, force=5, force_reset_time=200, link_name=None):
        assert force_info is None, NotImplementedError
        assert link_name is None, NotImplementedError

        self.force = force
        self.force_reset_time = force_reset_time
        self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
        self.force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
        self.cnt = 0
    
    def get_action(self):
        if self.cnt > self.force_reset_time:
            self.force_applied_force_vector = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) * self.force
            self.force_applied_position_vector = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.5)])
            self.cnt = 0
        self.cnt += 1
        return np.concatenate((self.force_applied_force_vector, self.force_applied_position_vector), axis=0)