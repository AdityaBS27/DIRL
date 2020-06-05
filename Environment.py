from DIRL.GAN import Generator , Discriminator
import tensorflow as tf 
from DIRL.Agent import Agent

class Environment():
    '''
    Caluclates the total rewards for current and loss at every step
    '''
    
    
    def __init__(self , sess,B, L):
        self.discriminator = Discriminator(sess, F)
        self.generator_mc = Generator(B, L,  E, H ,sess)
        self.g_mc = Agent(sess, B,L,0.001)
        self.T = 55
        self._sample = 10
        self.B = B
        self.L = L

    def step(self,cell_state,previous_action):
        self.t = self.t+1
        action, cell_state = self.g_mc.act_on_previous_state(cell_state, previous_action)
       
        action_Q = action
        cell_state_Q = cell_state

        reward = self.Q(cell_state_Q,action_Q,self._sample)
       
      
        episode_end = self.t>self.T 
        return  action, reward , episode_end 
        
    def Q(self,cell_state_it,previous_action_it,sample):
        reward =np.zeros([self.B,1])
        for i in range(sample):
          
            sample_future_trajectories = previous_action_it
            cell_state = cell_state_it
            previous_action = previous_action_it
            
            for tau in range(self.t+1 , self.T):
                action, state = self.g_mc.act_on_previous_state(cell_state, previous_action)
                sample_future_trajectories = np.concatenate([sample_future_trajectories, action],axis=-1)
                cell_state = state
                previous_action = action
            
            
            traj_reward = sample_future_trajectories.reshape(self.B,-1,1)
            reward += np.mean(np.asarray(self.discriminator.predict(traj_reward)).reshape(self.B,-1,1),axis =1).reshape(self.B,1)/sample
            
        return reward
      

    def reset(self):
         self.t = 1
