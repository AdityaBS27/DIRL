
from DIRL.GAN import Generator , Discriminator
from DIRL.Environment import Environment
from DIRL.Agent import Agent
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,AutoMinorLocator

class Trainer():
    def __init__(self , sess, B, L, E, H, G_mc ,discriminator, training_data, testing_data):
        self.L = L 
        self.E = E 
        self.H = H 
        self.B = B #batch size
        self.d = 15
        self.g = 30
        self.sess = sess
        self.g_mc = G_mc
        self.agent = Agent(sess,B, L,0.001)
        self.env = Environment(sess,B, L)
        self.train_data = training_data 
        self.test_data = testing_data 

        
#####################################----extract---data---------####################################
        
    def extract_train_data(self):
        a =  np.random.randint(low=0, high=750, size=1).tolist()
        new_data = self.train_data[:,:,a]
        new_data[new_data < 0] = 0
        new_data = new_data.reshape(self.B,-1)
        x =new_data[:,:30].reshape(self.B,-1)
        x_last = np.clip(new_data[:,29:30],0,self.L).reshape(self.B,-1)
        y = np.clip(new_data[:,30:],0,self.L).reshape(self.B,-1)
        return x,x_last,y
    
    def extract_test_data(self):
        a = np.random.randint(low=0, high=250, size=1).tolist()
        new_data = self.test_data[:,:,a]
        new_data[new_data < 0] = 0
        new_data = new_data.reshape(self.B,-1)
        x =new_data[:,:30].reshape(self.B,-1)
        y = np.clip(new_data[:,30:],0,self.L).reshape(self.B,-1)
        return x,y
    
##########################################---plot generator parameters-------###############################
    
    def plot_metrics(self,Y,actions,reward,times):
      
        actions = actions[:,1:].reshape(3,-1)


        fig, axs = plt.subplots(1, 3, figsize=(15, 7),
                               constrained_layout=True)

        

        ax = axs[0]
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.plot(Y[0,:],label='ego')
        ax.legend(loc = 'upper left')
        ax.plot(Y[1,:], label = 'Surround')
        ax.legend(loc = 'upper left')
        ax.plot(Y[2,:], label = 'Difference')
        ax.legend(loc = 'upper left')
        ax.set_xlim(-5,60)
        ax.set_xlabel('Time x 0.08 s')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_title('Orginal Trajectory')
        ax.grid(True)
        ax.tick_params(which='major', length=5)
        fig.suptitle(' Feature = Lateral Velocity,  Generator training , Epoch: %i' %times, fontsize=16)

        ax = axs[1]
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.plot(actions[0,:],label='ego')
        ax.legend(loc = 'upper left')
        ax.plot(actions[1,:], label = 'Surround')
        ax.legend(loc = 'upper left')
        ax.plot(actions[2,:], label = 'Difference')
        ax.legend(loc = 'upper left')
        ax.set_xlim(-5,60)
        ax.set_xlabel('Time x 0.08 s')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_title('Predicted Trajectory')
        ax.grid(True)
        ax.tick_params(which='major', length=5)
        
        
        ax = axs[2]
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.plot(reward[0,:],label='ego')
        ax.legend(loc = 'upper left')
        ax.plot(reward[1,:], label = 'Surround')
        ax.legend(loc = 'upper left')
        ax.plot(reward[2,:], label = 'Difference')
        ax.legend(loc = 'upper left')
        ax.set_xlabel('Cycles per epoch')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_title('Average Reward ')
        ax.grid(True)
        ax.tick_params(which='major', length=5)
        plt.show()
        return plt.show()
    
    
    def policy_plot(self,tensor):
        ego = tensor[0,1:,:].reshape(-1,self.L)
        follow = tensor[1,1:,:].reshape(-1,self.L)
        difference = tensor[2,1:,:].reshape(-1,self.L)

        x = np.arange(0,self.L,1)
        y = np.arange(0,55,1)
        Y, X = np.meshgrid(x,y)

        fig1 = plt.figure(figsize=(10,5))
        ax = fig1.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, ego, label=' policy')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_xlabel('Time x 0.08 s')
        ax.set_zlabel('Policy $\pi_{\theta}(s_t)$')
        ax.set_title('Ego policy')
        ax.view_init(40, 220)
        fig1.tight_layout()

        fig2 = plt.figure(figsize=(10,5))
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, follow, label=' policy')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_xlabel('Time x 0.08 s')
        ax.set_zlabel('Policy $\pi_{\theta}(s_t)$')
        ax.set_title('Surround policy')
        ax.view_init(40, 220)
        fig2.tight_layout()


        fig3 = plt.figure(figsize=(10,5))
        ax = fig3.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, difference, label=' policy')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_xlabel('Time x 0.08 s')
        ax.set_zlabel('Policy $\pi_{\theta}(s_t)$')
        ax.set_title('Difference policy')
        ax.view_init(40, 220)
        fig3.tight_layout()

        return fig1,fig2,fig3
    
    
##########################################---plot discriminator  parameters-------###############################     

    def sample_plot(self,Y,Y_hat,cycleloss,times):
        fig, axst = plt.subplots(1, 3, figsize=(15, 7),
                               constrained_layout=True)

        

        axp = axst[0]
        axp.xaxis.set_major_locator(MultipleLocator(10))
        axp.xaxis.set_minor_locator(MultipleLocator(5))
        axp.plot(Y[0,:],label='Ego')
        axp.legend(loc = 'upper left')
        axp.plot(Y[1,:], label = 'Surround')
        axp.legend(loc = 'upper left')
        axp.plot(Y[2,:], label = 'Difference')
        axp.legend(loc = 'upper left')
        axp.set_xlim(-5,60)
        axp.set_xlabel('Time x 0.08 s')
        axp.set_ylabel('Lateral velocity x 0.01m/s')
        axp.set_title('Orginal Trajectory')
        axp.grid(True)
        axp.tick_params(which='major', length=5)
        fig.suptitle(' Feature = Lateral Velocity, Discriminator training  Epoch: {}'.format(times) , fontsize=16)

        axp = axst[1]
        axp.xaxis.set_major_locator(MultipleLocator(10))
        axp.xaxis.set_minor_locator(MultipleLocator(5))
        axp.plot(Y_hat[0,:],label='Ego')
        axp.legend(loc = 'upper left')
        axp.plot(Y_hat[1,:], label = 'Surround')
        axp.legend(loc = 'upper left')
        axp.plot(Y_hat[2,:], label = 'Difference')
        axp.legend(loc = 'upper left')
        axp.set_xlim(-5,60)
        axp.set_xlabel('Time x 0.08 s')
        axp.set_ylabel('Lateral velocity x 0.01m/s')
        axp.set_title('Sampled trajectory')
        axp.grid(True)
        axp.tick_params(which='major', length=5)
        
        
        axp = axst[2]
        axp.xaxis.set_major_locator(MultipleLocator(10))
        axp.xaxis.set_minor_locator(MultipleLocator(5))
        axp.plot(cycleloss.T)
        axp.set_xlabel('Cycles per epoch')
        axp.set_ylabel('Lateral velocity x 0.01m/s')
        axp.set_title('Discriminator loss')
        axp.grid(True)
        axp.tick_params(which='major', length=5)

        plt.show()

        return plt.show()
    
      
    def policy_discriminator_plot(self,tensor):
  
        ego = tensor[0,:,:].reshape(-1,self.L)
        follow = tensor[1,:,:].reshape(-1,self.L)
        difference = tensor[2,:,:].reshape(-1,self.L)

        x = np.arange(0,self.L,1)
        y = np.arange(0,55,1)
        Y, X = np.meshgrid(x,y)


        fig1 = plt.figure(figsize=(10,5))
        ax = fig1.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, ego, label=' policy')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_xlabel('Time x 0.08 s')
        ax.set_zlabel(r'Policy $\pi_{\theta}(s_t)$')
        ax.set_title('Ego policy')
        ax.view_init(40, 220)
        fig1.tight_layout()

        fig2 = plt.figure(figsize=(10,5))
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, follow, label=' policy')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_xlabel('Time x 0.08 s')
        ax.set_zlabel('Policy $\pi_{\theta}(s_t)$')
        ax.set_title('Surround policy')
        ax.view_init(40,220)
        fig2.tight_layout()

        fig3 = plt.figure(figsize=(10,5))
        ax = fig3.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, difference, label=' policy')
        ax.set_ylabel('Lateral velocity x 0.01m/s')
        ax.set_xlabel('Time x 0.08 s')
        ax.set_zlabel('Policy $\pi_{\theta}(s_t)$')
        ax.set_title('Difference policy')
        ax.view_init(40, 220)
        fig3.tight_layout()

        return fig1,fig2,fig3
    
##########################################---plot test parameters-------###############################
      
    def plot_test(self,Y,Y_hat):
        fig, axst = plt.subplots(1, 2, figsize=(15, 7),
                               constrained_layout=True)


        axp = axst[0]
        axp.xaxis.set_major_locator(MultipleLocator(10))
        axp.xaxis.set_minor_locator(MultipleLocator(5))
        axp.plot(Y[0,:],label='Ego')
        axp.legend(loc = 'upper left')
        axp.plot(Y[1,:], label = 'Surround')
        axp.legend(loc = 'upper left')
        axp.plot(Y[2,:], label = 'Difference')
        axp.legend(loc = 'upper left')
        axp.set_xlim(-5,60)
        axp.set_xlabel('Time x 0.08 s')
        axp.set_ylabel('Lateral velocity x 0.01m/s')
        axp.set_title('Orginal Trajectory')
        axp.grid(True)
        axp.tick_params(which='major', length=5)
        fig.suptitle(' Test')

        axp = axst[1]
        axp.xaxis.set_major_locator(MultipleLocator(10))
        axp.xaxis.set_minor_locator(MultipleLocator(5))
        axp.plot(Y_hat[0,:],label='Ego')
        axp.legend(loc = 'upper left')
        axp.plot(Y_hat[1,:], label = 'Surround')
        axp.legend(loc = 'upper left')
        axp.plot(Y_hat[2,:], label = 'Difference')
        axp.legend(loc = 'upper left')
        axp.set_xlim(-5,60)
        axp.set_xlabel('Time x 0.08 s')
        axp.set_ylabel('Lateral velocity x 0.01m/s')
        axp.set_title('Sampled trajectory')
        axp.grid(True)
        axp.tick_params(which='major', length=5)
        
        plt.show()

        return plt.show()

###########################################################################################    
      
    def rmse(self, predicted, original):
        return np.sqrt(((predicted - original) ** 2).mean())
      
      
      
    def train(self, inital_steps, steps):
        
      
         ##################-------------Pre trainin-----------------##########################################
        
        
        for pre in range(inital_steps):
            

            X, X_last, Y = self.extract_train_data()

            Y_or = Y.reshape(self.B,-1,1)
            pos = np.ones_like(Y_or)*0.9

            for gen in range(50):
                
                pred = self.agent.act_on_input(X.astype(int))
                l = self.g_mc.encoder_training(X.astype(int),X.reshape(self.B,-1))


            initial_cell_state = self.agent.generator.input_cell_state
            future_traj = np.zeros([self.B,1])

            for j in range(55):
                
                if j == 0:
                    cell_state = initial_cell_state
                    previous_action = to_categorical(X_last,self.L).reshape(self.B,-1,self.L)

                future_prob, next_states,future_track, loss = self.agent.generator.update_decoder_crossentropy(cell_state, previous_action ,Y[:,j].reshape(self.B,-1))

                
                cell_states = next_states
                previous_action = future_prob

                
                future_traj = np.concatenate([future_traj,future_track],axis =-1)

            Y_hat = future_traj[:,1:].reshape(self.B,-1,1)
            neg =  np.zeros_like(Y_hat)


            combine = np.concatenate([Y_or,Y_hat], axis = 0)
            labels = np.concatenate([pos,neg], axis = 0)


            l = self.env.discriminator.train(combine,labels)
        #######################################################################
        
        
        
        total_discriminator_loss = np.zeros([1,steps])
        epoch_reward = np.zeros([1,steps])
        count = 0
        
        
        for iterate in range(steps):
        
            plot_fifty_epoch = np.arange(0,steps+1,50)
            inner_reward = np.zeros([self.B,1])
            
            #generator training
            for g_loop in range(self.g):
                X, X_last, Y = self.extract_train_data()

                for gen in range(25):
                
                    pred = self.agent.act_on_input(X.astype(int))
                    l = self.g_mc.encoder_training(X.astype(int),X.reshape(self.B,-1))
              
                self.env.reset()

                initial_cell_state = self.agent.generator.input_cell_state
                initial_action = self.agent.generator.input_action 

                rewards = np.zeros([self.B, 55])
                actions = np.zeros([self.B, 1], dtype=np.int32)
                policy_tensor = np.zeros([self.B,1,self.L])
            
            
                for i in range(55):

                    if i == 0:
                        temp_cell_state = initial_cell_state
                        temp_previous_action = initial_action


                    action,reward,episode = self.env.step(temp_cell_state, temp_previous_action)
                    future_prob , next_states, loss = self.g_mc.update_decoder(temp_cell_state, temp_previous_action, action, reward)


                    policy_tensor = np.hstack((policy_tensor, future_prob))

                    _,_ ,_, _ = self.agent.generator.update_decoder_crossentropy(temp_cell_state, temp_previous_action ,Y[:,i].reshape(self.B,-1))

                    actions = np.concatenate([actions, action], axis = -1)
                    rewards[:, i] = reward.reshape([self.B, ])

                    temp_cell_state = next_states
                    temp_previous_action= future_prob


                    if episode == True :
                        inner_reward = np.concatenate([inner_reward,rewards.mean(axis = 1).reshape(self.B,-1)],axis =-1)

            if iterate in plot_fifty_epoch:
                self.plot_metrics(Y,actions,inner_reward, count)
                #if iterate == 0 or iterate==steps-1 :
                    #self.policy_plot(policy_tensor)
                    
             
            epoch_reward[:,[iterate]] = inner_reward.mean().reshape(1,1)
                
          
              ##########################-discriminator training---##########################
            
            inner_discrimintor_loss  = np.zeros([1,self.g])
            
            for d_loop in range(self.g):
                

                X, X_last, Y = self.extract_train_data()
                pred = self.agent.act_on_input(X.astype(int))
                

                for dis in range(self.d):
                    self.agent.reset()

                    cell_state = self.agent.generator.input_cell_state
                    previous_action = self.agent.generator.input_action


                    Y = Y.reshape(self.B,-1,1)

                    Y_hat, policy_discriminator = self.g_mc.sample_trajectories(55, cell_state, previous_action)
            


                    neg_s = np.zeros_like(Y_hat)
                    pos_s = np.ones_like(Y)


                    labeles_s = np.concatenate([pos_s, neg_s],axis = 0)
                    combine_s = np.concatenate([Y, Y_hat] ,axis = 0) 


                    discrimintor_training_loss = self.env.discriminator.train(combine_s,labeles_s).mean().reshape(1,-1)

                    if dis == self.d-1:
                        inner_discrimintor_loss[0,[d_loop]] = discrimintor_training_loss                      
                        self.inner_discrimintor_loss = inner_discrimintor_loss
           
            if iterate in plot_fifty_epoch:
                self.sample_plot(Y,Y_hat,inner_discrimintor_loss,count)
                #if iterate == 0 or iterate==steps-1 :
                    #self.policy_discriminator_plot(policy_discriminator)
                    

            total_discriminator_loss[:,[iterate]] = inner_discrimintor_loss.mean().reshape(1,1)    
            
            count = count+1

            self.rewards = epoch_reward 
            self.d_loss = total_discriminator_loss
        

            
        
        
    def test(self, samples) :
        rmse_test = np.zeros([1,samples])
        
        for k in range(samples):
            
            X,Y = self.extract_test_data()
            s = np.arange(0,samples,20)

            self.agent.act_on_input(X)


            cell_state = self.agent.generator.input_cell_state
            previous_action = self.agent.generator.input_action


            Y = Y.reshape(self.B,-1,1)

            Y_hat, test_policy_discriminator = self.g_mc.sample_trajectories(55, cell_state, previous_action)


            rmse_test[:,[k]] = self.rmse(Y,Y_hat).reshape(1,1)

            if k in s:
                self.plot_test(Y,Y_hat)

        self.error = rmse_test
