
from DIRL.GAN import Generator 

class Agent():
    
    '''
    On each step, Agent act on previous trajectories.
    Then Environment return next state, reward, and so on.
    '''
    def __init__(self, sess, B, L,lr):
        
        self.sess = sess
        self.num_actions = L
        self.B = B
        self.L = L
        self.lr = lr
        self.eps = 0.1
        self.generator = Generator(B, L,  E, H ,sess)
        
    
    def act_on_input(self,data):
        return  self.generator.predict_encoder(data)

    
    def act_on_previous_state(self, cell_state, previous_action):
        vary = np.random.rand()
        
        if  vary <= self.eps :
            probs, next_cell_state = self.generator.predict_next_state(cell_state,previous_action)
            action = self.generator.select_random_action(probs)
        else :
            probs, next_cell_state = self.generator.predict_next_state(cell_state,previous_action)
            action = np.argmax(probs, axis=-1).reshape([self.B, 1])

        def store_next_set(self):
            self.store_next_probs = probs
            self.store_next_action = action
            self.store_next_cell_state = next_cell_state 
        
        store_next_set(self)
        
        return action, next_cell_state 
    

    def reset(self):
        self.generator.input_cell_state 
        self.generator.encoder_prediction 
        self.generator.input_action
        self.generator.input

