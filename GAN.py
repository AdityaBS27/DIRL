import os
import tensorflow as tf
from numpy import argmax
from keras import objectives, backend as K
from tensorflow.keras.utils import to_categorical


class Generator():
    
    def __init__(self,B, L,  E, H ,sess):

        self.L = L 
        self.E = E 
        self.H = H 
        self.B = B 
        self.sess = sess
        self.build_generator_graph()




    def build_generator_graph(self):
      
        tf.random.set_random_seed(123)
        ######################use same cells for encoder and decoder ###############
        reuse = tf.AUTO_REUSE


      ##########################-------------encode-------------r########################################
        with tf.variable_scope('encoder',reuse =  tf.AUTO_REUSE ):
            X = tf.placeholder(tf.int32, shape = (self.B,None))
            X_hat = tf.placeholder(tf.float32, shape = (self.B,None,self.L))
            e_c = tf.nn.rnn_cell.LSTMCell(self.H,state_is_tuple=True, reuse = reuse)
            Y = tf.Variable(tf.random_uniform([self.L,self.E], -1.0, 1.0),trainable=True, dtype=tf.float32)
            encoder_in = tf.nn.embedding_lookup(Y,X)
            hist_pred, hist_state = tf.nn.dynamic_rnn(e_c, encoder_in,  dtype=tf.float32,time_major=False,)
            hist_logit = tf.layers.dense(inputs=hist_pred, units= self.L, activation=tf.nn.softmax)
            hist_traj = tf.reshape(tf.argmax(hist_logit,2),[self.B,-1,1])
            enc_loss = -tf.reduce_sum(X_hat * tf.log(hist_logit))

        

      
      
        with tf.variable_scope('encoder_loss', reuse =  tf.AUTO_REUSE):
            enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="encoder")
            train_encoder = tf.train.AdamOptimizer(0.001).minimize(enc_loss, var_list = enc_vars)
          



      #######################---------------#decode---------------------r#########################################
        with tf.variable_scope("decoder",reuse = tf.AUTO_REUSE):
            
            Y_hat = tf.placeholder(tf.float32, shape = (self.B,None,self.L))
            reward = tf.placeholder(tf.float32, shape = (self.B, None))
            action = tf.placeholder(tf.float32, shape = (self.B,1, self.L))
            previous_traj_state = tf.placeholder(tf.float32, shape = (self.B,None,self.L))
            state_c = tf.placeholder(tf.float32, [self.B, None], name="initial_lstm_state_c")
            state_h = tf.placeholder(tf.float32, [self.B, None], name="initial_lstm_state_h")
            state_in =tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h)
            

            d_c = tf.nn.rnn_cell.LSTMCell(self.H,state_is_tuple=True, reuse = reuse)
            future_prediction , next_state = tf.nn.dynamic_rnn(d_c,  previous_traj_state, initial_state = state_in ,time_major=False,)
            future_prob = tf.layers.dense(inputs=future_prediction,units = self.L, activation=tf.nn.softmax)
            future_trajectories = tf.clip_by_value(tf.argmax(future_prob,2),0, self.L)
            log_prob = tf.log(tf.reduce_mean(future_prob * action, axis=-1))
            r_loss = - log_prob * reward
            complete_loss = -tf.reduce_sum(Y_hat * tf.log(future_prob))
            

        with tf.variable_scope('reward_loss', reuse = tf.AUTO_REUSE):
            reward_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="decoder")
            minimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(r_loss, var_list =reward_vars )   


        with tf.variable_scope('complete_loss', reuse = tf.AUTO_REUSE):
            cross_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="decoder")
            train = tf.train.AdamOptimizer(0.001).minimize(complete_loss , var_list =cross_vars )
            

      #############################################################################


          #input variable    
        self.action = action
        self.reward = reward
        self.X = X
        self.X_hat = X_hat
        self.Y_hat =Y_hat

        #encoder variable
        self.embedding_matrix = encoder_in
        self.enc_out = hist_traj
        self.hist_pred = hist_pred
        self.decoder_input = hist_logit
        self.hist_state = hist_state
        self.hist_traj = hist_traj 

        self.encoder_loss = enc_loss
        self.encoder_train = train_encoder
     
      
      
      #decoder variable
        self.state_in = state_in
        self.feed_traj_state = previous_traj_state
        self.future_prob = future_prob
        self.future_states = next_state
        self.future_traj = future_trajectories



      #decoder_loss
        self.loss = r_loss
        self.minimize = minimize

      #decoder encoder loss
        self.complete_loss = complete_loss 
        self.train = train
        
      #############
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

  
      #######################################------------predict---and---update----------------------####################################################


       ##########################-------------encoder trianing-------------r########################################    
     
    def predict_encoder(self,data , reuse = False):
        
    #X#
        his_prediction, input_cell_state, traj = self.sess.run([self.decoder_input,self.hist_state, self.hist_traj],
                                                           feed_dict = {self.X: data })

        def inital_state_action(self):
            self.input_cell_state = input_cell_state
            self.encoder_prediction = his_prediction
            self.input_action = his_prediction[:,29,:].reshape(self.B,-1,self.L)
            self.input = traj  

        inital_state_action(self)

        return his_prediction 


 
    def encoder_training(self,data, data_labels , reuse = True):
        loss , _ = self.sess.run([self.encoder_loss , self.encoder_train], feed_dict = {self.X: data,
                                                                                  self.X_hat : to_categorical(np.clip(data_labels,0,self.L-5), self.L).reshape(self.B,-1,self.L) })

        return loss


  
   ##########################-------------decoder  trainig-------------r########################################

    def predict_next_state(self,lstm_cell_state , previous_action , reuse = False): 

        if previous_action.ndim == 2:
            
            previous_action = to_categorical(previous_action,self.L).reshape(self.B,-1,self.L)
        else:
            
            previous_action = previous_action

        future_prob,next_cell_state = self.sess.run([self.future_prob,self.future_states], 
                                                         feed_dict = {self.feed_traj_state: previous_action,
                                                                      self.state_in : lstm_cell_state})

        return future_prob,next_cell_state




    def update_decoder(self, state_in, previous_prob, action, reward, reuse =True):
        


        feed_dict = {self.state_in : state_in, #tuple
                   self.feed_traj_state: previous_prob.reshape(self.B,-1,self.L), #  (B,1, L)
                   self.action : to_categorical(action, self.L).reshape(self.B,-1,self.L),#(B,1,L)
                   self.reward : reward #(B,1)
                    }
                     
        future_prob, next_states, loss,_  = self.sess.run([self.future_prob,self.future_states, self.loss ,self.minimize ],feed_dict)

        return  future_prob , next_states ,loss

  
  
    def update_decoder_crossentropy(self, state_in, previous_prob, data, reuse=True):
        
                     
        future_prob, next_states,track, loss, _ = self.sess.run([self.future_prob, self.future_states,self.future_traj, self.complete_loss, self.train],
                                                        feed_dict ={self.state_in : state_in, 
                                                       self.feed_traj_state: previous_prob,
                                                       self.Y_hat : to_categorical(np.clip(data,0, self.L-5), self.L).reshape(self.B,-1,self.L)})
        return  future_prob, next_states ,track, loss


  ##########################-------------sample prediction-------------r##################################################
  
    def select_random_action(self,prob): #X#
        
        action = np.zeros((self.B,), dtype=np.int32)
        for i in range(self.B):
            p = prob[i].reshape(-1)
            p[np.isnan(p)] = 0
            action[i] = np.random.choice(self.L, p=p)
        return action.reshape(-1,1)

  
    def sample_trajectories(self, T , cell_state, previous_action, reuse = False):
        action = np.zeros([self.B, 1], dtype=np.int32)
        logit_decoder = np.zeros([self.B, 1 , self.L], dtype=np.int32)
        prob_trajectories = logit_decoder
        actions = action

        sample_cell_state = cell_state
        sample_previous_action = previous_action

        for _ in range(T):
            prob , next_state = self.predict_next_state(sample_cell_state,sample_previous_action)
            action = self.select_random_action(prob)

            actions = np.concatenate([actions, action], axis=-1)
            prob_trajectories = np.concatenate([prob_trajectories, prob], axis=1)

            sample_cell_state = next_state
            sample_previous_action = to_categorical(action, self.L).reshape(self.B,-1, self.L)


        actions = actions[:, 1:]
        prob_trajectories = prob_trajectories[:, 1: ,:]


        return actions.reshape(self.B,-1,1) , prob_trajectories



   #################


class Discriminator():
    
    def __init__(self, sess, F):
        self.F = F
        self.sess = sess
        self.build_discriminator_graph()
        
    def build_discriminator_graph(self):
        reuse = tf.AUTO_REUSE 

        with tf.variable_scope('d' , reuse = tf.AUTO_REUSE):
            d_X = tf.placeholder(tf.float32, shape = (None,None,self.F))
            d_X_hat = tf.placeholder(tf.float32, shape = (None,None,self.F))
            dis = tf.nn.rnn_cell.LSTMCell(1024)
            dis_prob, dis_state = tf.nn.dynamic_rnn(dis, d_X,  dtype=tf.float32,time_major=False,)
            dis_logit = tf.layers.dense(inputs=dis_prob, units= 1, activation=None)
            pred = tf.nn.sigmoid(dis_logit) 
            d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=d_X_hat, logits=pred),1)

    
        with tf.variable_scope('dpredict', reuse = tf.AUTO_REUSE):
            gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="d")
            trian_d = tf.train.AdamOptimizer(0.0001).minimize(d_loss,var_list = gen_vars) 


        self.input = d_X
        self.label = d_X_hat
        self.dis_prob = dis_prob
        self.dis_logit = dis_logit
        self.pred = pred 
        self._loss = d_loss
        self._train = trian_d 

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)


    
    def predict(self,data,reuse =False):
        out = self.sess.run([self.pred], feed_dict ={self.input : data})
        return out

      
    def train(self, data, target, reuse =True):
        loss, _ = self.sess.run([self._loss, self._train], feed_dict= {self.input : data, self.label :  target })
        return loss
