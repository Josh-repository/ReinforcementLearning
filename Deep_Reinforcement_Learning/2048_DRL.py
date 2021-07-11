import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
from copy import deepcopy
import random 
import math
from numpy import rot90
from numpy import array, zeros

class GameStructure:
    #Creating a new game board.

    def game_board(self,n):
        matrix = np.zeros([n,n])
        return matrix
    
    #Randomly adding new number in to the grid. 
    def generateNewNumber(self,grid):
        empty_cells = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if(grid[i][j]==0):
                    empty_cells.append((i,j))
        if(len(empty_cells)==0):
            return grid   
        index_pair = empty_cells[random.randint(0,len(empty_cells)-1)]
        prob = random.random()
        if(prob>=0.9):
            grid[index_pair[0]][index_pair[1]]=4
        else:
            grid[index_pair[0]][index_pair[1]]=2
        return grid

    #Checking whether the game reaches the goal state or not.
    #Checking for any possible move from current state.
    def CanMakeMove(self,grid):    
        for i in range(len(grid)-1): #intentionally reduced to check the row on the right and below
            for j in range(len(grid[0])-1): #more elegant to use exceptions but most likely this will be their solution
                if grid[i][j]==grid[i+1][j] or grid[i][j+1]==grid[i][j]:
                    return 'CanMakeMove'
            
        for i in range(len(grid)): #check for any zero entries
            for j in range(len(grid[0])):
                if grid[i][j]==0:
                    return 'CanMakeMove'
            
        for k in range(len(grid)-1): #to check the left/right entries on the last row
            if grid[len(grid)-1][k]==grid[len(grid)-1][k+1]:
                return 'CanMakeMove'
        
        for j in range(len(grid)-1): #check up/down entries on last column
            if grid[j][len(grid)-1]==grid[j+1][len(grid)-1]:
                return 'CanMakeMove'
        return 'Game Lost'

    def isGameWon(self,grid):
        for i in range(len(grid)): #check for any entries with 2048 tile.
            for j in range(len(grid[0])):
                if grid[i][j]==2048:
                    return 'Game Won'
                else:
                    result = self.CanMakeMove(grid)
                    return result
    
    # Defining moves to perform grid transformation.
    def reverse(self,grid):
        new=[]
        for i in range(len(grid)):
            new.append([])
            for j in range(len(grid[0])):
                new[i].append(grid[i][len(grid[0])-j-1])
        return new

    def transpose(self,grid):
        new=[]
        for i in range(len(grid[0])):
            new.append([])
            for j in range(len(grid)):
                new[i].append(grid[j][i])
                
        return np.transpose(grid)

    def cover_up(self,grid):
        new = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        done = False
        for i in range(4):
            count = 0
            for j in range(4):
                if grid[i][j]!=0:
                    new[i][count] = grid[i][j]
                    if j!=count:
                        done=True
                    count+=1
        return (new,done)

    def merge(self,grid):
        done=False
        score = 0
        for i in range(4):
            for j in range(3):
                if grid[i][j]==grid[i][j+1] and grid[i][j]!=0:
                    grid[i][j]*=2
                    score += grid[i][j]   
                    grid[i][j+1]=0
                    done=True
        return (grid,done,score)

    #up move
    def up(self,game):
            game = self.transpose(game)
            game,done = self.cover_up(game)
            temp = self.merge(game)
            game = temp[0]
            done = done or temp[1]
            game = self.cover_up(game)[0]
            game = self.transpose(game)
            return (game,done,temp[2])

    #down move
    def down(self,game):
            game=self.reverse(self.transpose(game))
            game,done=self.cover_up(game)
            temp=self.merge(game)
            game=temp[0]
            done=done or temp[1]
            game=self.cover_up(game)[0]
            game=self.transpose(self.reverse(game))
            return (game,done,temp[2])

    #left move
    def left(self,game):
            game,done=self.cover_up(game)
            temp=self.merge(game)
            game=temp[0]
            done=done or temp[1]
            game=self.cover_up(game)[0]
            return (game,done,temp[2])

    #right move
    def right(self,game):
            game=self.reverse(game)
            game,done=self.cover_up(game)
            temp=self.merge(game)
            game=temp[0]
            done=done or temp[1]
            game=self.cover_up(game)[0]
            game=self.reverse(game)
            return (game,done,temp[2])
    controls = {0:up,1:left,2:right,3:down}
    print ("Successfully modified the game play")
    
class Tensorflow(GameStructure):
    #convert the input game matrix into corresponding power of 2 matrix.
    def __init__(self):
        
        self.controls = super().controls
        #gradient decent learning hyper parameter.
        self.learning_rate = 0.0001

        #gamma for Q-learning
        self.gamma = 0.8

        #epsilon greedy approach
        self.epsilon = 0.8

        #to store states and lables of the game for training
        #states of the game
        self.state_memory = list()

        #labels of the states
        self.q_labels = list()

        #capacity of memory
        self.memory = 7000
        #loss
        self.loss = []

        #scores
        self.scores = []

        #to store final parameters
        self.final_parameters = {}

        #number of episodes
        self.epochs = 180000
        
        #first and second convolution layer depth
        d1 = 128 
        d2 = 128

        #batch size for batch gradient descent
        self.batch_size = 512

        #input units
        input_units = 16

        #fully connected layer neurons
        hidden_units = 256

        #output neurons = number of moves
        self.output_units = 4
        
        self.max_tile = []
        self.score = []
        #layer1
        #CONV LAYERS
        #conv layer1 weights
        self.conv1_layer1_weights = tf.Variable(tf.truncated_normal([1,2,input_units,d1],mean=0,stddev=0.01))
        self.conv2_layer1_weights = tf.Variable(tf.truncated_normal([2,1,input_units,d1],mean=0,stddev=0.01))

        #conv layer2 weights
        self.conv1_layer2_weights = tf.Variable(tf.truncated_normal([1,2,d1,d2],mean=0,stddev=0.01))
        self.conv2_layer2_weights = tf.Variable(tf.truncated_normal([2,1,d1,d2],mean=0,stddev=0.01))



        #FUllY CONNECTED LAYERS
        expand_size = 2*4*d2*2 + 3*3*d2*2 + 4*3*d1*2
        self.fc_layer1_weights = tf.Variable(tf.truncated_normal([expand_size,hidden_units],mean=0,stddev=0.01))
        self.fc_layer1_biases = tf.Variable(tf.truncated_normal([1,hidden_units],mean=0,stddev=0.01))
        self.fc_layer2_weights = tf.Variable(tf.truncated_normal([hidden_units,self.output_units],mean=0,stddev=0.01))
        self.fc_layer2_biases = tf.Variable(tf.truncated_normal([1,self.output_units],mean=0,stddev=0.01))       
    
    def tensor_rep(self,grid):
        power_grid = np.zeros(shape=(1,4,4,16),dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if(grid[i][j]==0):
                    power_grid[0][i][j][0] = 1.0
                else:
                    power = int(math.log(grid[i][j],2))
                    power_grid[0][i][j][power] = 1.0
        return power_grid        

    #find the number of empty cells in the game matrix.
    def findemptyCell(self,grid):
        iterator = 0
        for i in range(len(grid)):
            for j in range(len(grid)):
                if(grid[i][j]==0):
                    iterator+=1
        return iterator

    #network
    def network(self,dataset):        
        #Network creation
        conv1 = tf.nn.conv2d(dataset,self.conv1_layer1_weights,[1,1,1,1],padding='VALID') 
        conv2 = tf.nn.conv2d(dataset,self.conv2_layer1_weights,[1,1,1,1],padding='VALID') 
        
        #layer1 relu activation
        relu1 = tf.nn.relu(conv1)
        relu2 = tf.nn.relu(conv2)
        
        #layer2
        conv11 = tf.nn.conv2d(relu1,self.conv1_layer2_weights,[1,1,1,1],padding='VALID') 
        conv12 = tf.nn.conv2d(relu1,self.conv2_layer2_weights,[1,1,1,1],padding='VALID') 

        conv21 = tf.nn.conv2d(relu2,self.conv1_layer2_weights,[1,1,1,1],padding='VALID') 
        conv22 = tf.nn.conv2d(relu2,self.conv2_layer2_weights,[1,1,1,1],padding='VALID') 

        #layer2 relu activation
        relu11 = tf.nn.relu(conv11)
        relu12 = tf.nn.relu(conv12)
        relu21 = tf.nn.relu(conv21)
        relu22 = tf.nn.relu(conv22)
        
        #get shapes of all activations
        shape1 = relu1.get_shape().as_list()
        shape2 = relu2.get_shape().as_list()
        
        shape11 = relu11.get_shape().as_list()
        shape12 = relu12.get_shape().as_list()
        shape21 = relu21.get_shape().as_list()
        shape22 = relu22.get_shape().as_list()

        #expansion
        hidden1 = tf.reshape(relu1,[shape1[0],shape1[1]*shape1[2]*shape1[3]])
        hidden2 = tf.reshape(relu2,[shape2[0],shape2[1]*shape2[2]*shape2[3]])
        
        hidden11 = tf.reshape(relu11,[shape11[0],shape11[1]*shape11[2]*shape11[3]])
        hidden12 = tf.reshape(relu12,[shape12[0],shape12[1]*shape12[2]*shape12[3]])
        hidden21 = tf.reshape(relu21,[shape21[0],shape21[1]*shape21[2]*shape21[3]])
        hidden22 = tf.reshape(relu22,[shape22[0],shape22[1]*shape22[2]*shape22[3]])

        #concatenation
        hidden = tf.concat([hidden1,hidden2,hidden11,hidden12,hidden21,hidden22],axis=1)

        #full connected layers
        hidden = tf.matmul(hidden,self.fc_layer1_weights) + self.fc_layer1_biases
        hidden = tf.nn.relu(hidden)

        #output layer
        output = tf.matmul(hidden,self.fc_layer2_weights) + self.fc_layer2_biases
        
        #return output
        return output

    #Main Implementation part:
    def train_method(self):
        #input data
        tf_batch_dataset = tf.placeholder(tf.float32,shape=(self.batch_size,4,4,16))
        tf_batch_labels  = tf.placeholder(tf.float32,shape=(self.batch_size,self.output_units))
        single_dataset   = tf.placeholder(tf.float32,shape=(1,4,4,16))
        
        with tf.Session() as session:
            #keep track of episode with maximum score.
            #for single example
            single_output = self.network(single_dataset)

            #for batch data
            logits = self.network(tf_batch_dataset)
            #loss
            loss = tf.square(tf.subtract(tf_batch_labels,logits))
            loss = tf.reduce_sum(loss,axis=1,keep_dims=True)
            loss = tf.reduce_mean(loss)/2.0

            #optimizer
            global_step = tf.Variable(0)  # count the number of steps taken.
            learning_rate = tf.train.exponential_decay(float(self.learning_rate), global_step, 1000, 0.90, staircase=True)
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
            tf.global_variables_initializer().run()
            print("Initialized")
            maximum = -1
            episode = -1
            
            #total_iters 
            total_iters = 1
            
            #number of back props
            back=0

            for i in range(self.epochs):
                loop = i
                global board
                board = super().game_board(4)
                super().generateNewNumber(board)
                super().generateNewNumber(board)
                
                #whether episode finished or not
                finish = 'CanMakeMove'
                
                #total_score of this episode
                total_score = 0
                
                #iters per episode
                local_iters = 1

                while(finish=='CanMakeMove'):
                    prev_board = deepcopy(board)
                    
                    #get the required move for this state
                    state = deepcopy(board)
                    state = self.tensor_rep(state)
                    state = np.array(state,dtype = np.float32).reshape(1,4,4,16)
                    feed_dict = {single_dataset:state}
                    control_scores = session.run(single_output,feed_dict=feed_dict)
                    
                    #find the move with max Q value
                    control_buttons = np.flip(np.argsort(control_scores),axis=1)
                    
                    #copy the Q-values as labels
                    labels = deepcopy(control_scores[0])
                    
                    #generate random number for epsilon greedy approach
                    num = random.uniform(0,1)
                    
                    #store prev max
                    prev_max = np.max(prev_board)
                    
                    #num is less epsilon generate random move
                    if(num<self.epsilon):
                        #find legal moves
                        legal_moves = list()
                        for i in range(4):
                            temp_board = deepcopy(prev_board)
                            temp_board,_,_ = super().controls[i](self,temp_board)
                            if(np.array_equal(temp_board,prev_board)):
                                continue
                            else:
                                legal_moves.append(i)
                        if(len(legal_moves)==0):
                            finish = 'Game Over'
                            continue
                        
                        #generate random move.
                        con = random.sample(legal_moves,1)[0]
                        
                        #apply the move
                        temp_state = deepcopy(prev_board)
                        temp_state,_,score = super().controls[con](self,temp_state)
                        total_score += score
                        finish = super().isGameWon(temp_state)
                        
                        #get number of merges
                        empty1 = self.findemptyCell(prev_board)
                        empty2 = self.findemptyCell(temp_state)
                        
                        if(finish=='CanMakeMove'):
                            temp_state = super().generateNewNumber(temp_state)

                        board = deepcopy(temp_state)

                        #get next max after applying the move
                        next_max = np.max(temp_state)
                        
                        #reward math.log(next_max,2)*0.1 if next_max is higher than prev max
                        labels[con] = (math.log(next_max,2))*0.1
                        
                        if(next_max==prev_max):
                            labels[con] = 0
                        
                        #reward is also the number of merges
                        labels[con] += (empty2-empty1)
                        
                        #get the next state max Q-value
                        temp_state = self.tensor_rep(temp_state)
                        temp_state = np.array(temp_state,dtype = np.float32).reshape(1,4,4,16)
                        feed_dict = {single_dataset:temp_state}
                        temp_scores = session.run(single_output,feed_dict=feed_dict)
                            
                        max_qvalue = np.max(temp_scores)
                        
                        #final labels add gamma*max_qvalue
                        labels[con] = (labels[con] + self.gamma*max_qvalue)
                    
                    #generate the max predicted move
                    else:
                        for moves in control_buttons[0]:
                            prev_state = deepcopy(prev_board)
                            
                            #apply the LEGAl Move with max q_value
                            temp_state,_,score = super().controls[moves](self,prev_state)
                            
                            
                            #if legal move label = 0
                            if(np.array_equal(prev_board,temp_state)):
                                labels[moves] = 0
                                continue
                                
                            #get number of merges
                            empty1 = self.findemptyCell(prev_board)
                            empty2 = self.findemptyCell(temp_state)

                            
                            temp_state = super().generateNewNumber(temp_state)
                            board = deepcopy(temp_state)
                            total_score += score

                            next_max = np.max(temp_state)
                            
                            #reward
                            labels[moves] = (math.log(next_max,2))*0.1
                            if(next_max==prev_max):
                                labels[moves] = 0
                            
                            labels[moves] += (empty2-empty1)

                            #get next max qvalue
                            temp_state = self.tensor_rep(temp_state)
                            temp_state = np.array(temp_state,dtype = np.float32).reshape(1,4,4,16)
                            feed_dict = {single_dataset:temp_state}
                            temp_scores = session.run(single_output,feed_dict=feed_dict)

                            max_qvalue = np.max(temp_scores)

                            #final labels
                            labels[moves] = (labels[moves] + self.gamma*max_qvalue)
                            break
                            
                        if(np.array_equal(prev_board,board)):
                            finish = 'Game Over'
                    
                    #decrease the epsilon value
                    if((i>10000) or (self.epsilon>0.1 and total_iters%2500==0)):
                        self.epsilon = self.epsilon/1.005
                        
                
                    #change the matrix values and store them in memory
                    prev_state = deepcopy(prev_board)
                    prev_state = self.tensor_rep(prev_state)
                    prev_state = np.array(prev_state,dtype=np.float32).reshape(1,4,4,16)
                    self.q_labels.append(labels)
                    self.state_memory.append(prev_state)
                    
                    #back-propagation
                    if(len(self.state_memory)>=self.memory):
                        back_loss = 0
                        batch_num = 0
                        z = list(zip(self.state_memory,self.q_labels))
                        np.random.shuffle(z)
                        np.random.shuffle(z)
                        self.state_memory,self.q_labels = zip(*z)
                        
                        for i in range(0,len(self.state_memory),self.batch_size):
                            if(i + self.batch_size>len(self.state_memory)):
                                break
                                
                            batch_data = deepcopy(self.state_memory[i:i+self.batch_size])
                            batch_labels = deepcopy(self.q_labels[i:i+self.batch_size])
                            
                            batch_data = np.array(batch_data,dtype=np.float32).reshape(self.batch_size,4,4,16)
                            batch_labels = np.array(batch_labels,dtype=np.float32).reshape(self.batch_size,self.output_units)
                        
                            feed_dict = {tf_batch_dataset: batch_data,tf_batch_labels: batch_labels}
                            _,l = session.run([optimizer,loss],feed_dict=feed_dict)
                            back_loss += l 
                            
                            print("Mini-Batch - {} Back-Prop : {}, Loss : {}".format(batch_num,back,l))
                            batch_num +=1
                        back_loss /= batch_num
                        self.loss.append(back_loss)
                        
                        #store the parameters in a dictionary
                        self.final_parameters['conv1_layer1_weights'] = session.run(self.conv1_layer1_weights)
                        self.final_parameters['conv1_layer2_weights'] = session.run(self.conv1_layer2_weights)
                        self.final_parameters['conv2_layer1_weights'] = session.run(self.conv2_layer1_weights)
                        self.final_parameters['conv2_layer2_weights'] = session.run(self.conv2_layer2_weights)
                        self.final_parameters['fc_layer1_weights'] = session.run(self.fc_layer1_weights)
                        self.final_parameters['fc_layer2_weights'] = session.run(self.fc_layer2_weights)
                        self.final_parameters['fc_layer1_biases'] = session.run(self.fc_layer1_biases)
                        self.final_parameters['fc_layer2_biases'] = session.run(self.fc_layer2_biases)
                        
                        #number of back-props
                        back+=1
                        
                        #make new memory 
                        self.state_memory = list()
                        self.q_labels = list()
                        
                    
                    if(local_iters%400==0):
                        print("Episode : {}, Score : {}, Iters : {}, Finish : {}".format(i,total_score,local_iters,finish))
                    
                    local_iters += 1
                    total_iters += 1  
                self.scores.append(total_score)
                temp = 0
                for i in range(len(board)):
                    for j in range(len(board)):
                        if(board[i][j] > temp):
                            temp = board[i][j]
                self.max_tile.append(temp)
                self.score.append(total_score)
                print("Episode {} finished with score {}, result : {} board : {}, maximum tile value: {}, epsilon  : {}, learning rate : {} ".format(loop,total_score,finish,board,temp,self.epsilon,session.run(learning_rate)))
                print()
                
                if((i+1)%1000==0):
                    print("Maximum Score : {} ,Episode : {}".format(maximum,episode))    
                    print("Loss : {}".format(self.loss[len(self.loss)-1]))
                    print()
                    
                if(maximum<total_score):
                    maximum = total_score
                    episode = loop
            print("Maximum Score : {} ,Episode : {}".format(maximum,episode))
            path = r'Final_Weights'
            weights = ['conv1_layer1_weights','conv1_layer2_weights','conv2_layer1_weights','conv2_layer2_weights','fc_layer1_weights','fc_layer1_biases','fc_layer2_weights','fc_layer2_biases']
            for w in weights:
                flatten = self.final_parameters[w].reshape(-1,1)
                file = open(path + '\\' + w +'.csv','w')
                file.write('Sno,Weight\n')
                for i in range(flatten.shape[0]):
                    file.write(str(i) +',' +str(flatten[i][0])+'\n') 
                file.close()
                print(w + " written!")
        return self.max_tile,self.score,self.loss
x = Tensorflow()
a,b,c = x.train_method()
