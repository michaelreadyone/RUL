import numpy as np
import torch
import torch.nn as nn
from model_paper import Net
from model_diy import TransformerModel
from utils import evaluation, get_train_test, relative_error, setup_seed

# Load data
Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('datasets/CALCE/CALCE.npy', allow_pickle=True)
Battery = Battery.item()

# Train function
def train(Rated_Capacity, K, lr=0.01, feature_size=8, feature_num=1, hidden_dim=32, num_layers=1, nhead=1, dropout=0.0, epochs=1000, 
          weight_decay=0.0, seed=0, alpha=0.0, noise_level=0.0, metric='re', device='cpu'):

    score_list, fixed_result_list, moving_result_list = [], [], []
    setup_seed(seed)
    for i in range(4):
        name = Battery_list[i]
        print(f'name: {name}, feature_size: {feature_size}')
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, feature_size)
        test_sequence = train_data + test_data
        # print('sample size: {}'.format(len(train_x)))

        # model = Net(feature_size=feature_size, hidden_dim=hidden_dim, feature_num=K, num_layers=num_layers, 
        #             nhead=nhead, dropout=dropout, noise_level=noise_level)
        model = TransformerModel(input_dim=feature_size, d_model=32, nhead=1, num_layers=3, dropout=0)
        print(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_fixed_slice, y_moving_slice = [0], [], []
        rmse, re = 1, 1
        score_, score = [1],[1]
        for epoch in range(epochs):
            x = np.reshape(train_x/Rated_Capacity,(-1, feature_num, feature_size)).astype(np.float32)
            y = np.reshape(train_y/Rated_Capacity,(-1,1)).astype(np.float32) 

            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            x = x.repeat(1, K, 1)
            output = model(x)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)
            print(f'loss per epoch: {loss}')
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step()                   

            if (epoch + 1)%10 == 0:
                test_x = train_data.copy() 
                fixed_point_list, moving_point_list = [], []
                t = 0
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:])/Rated_Capacity,(-1, feature_num, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device) 
                    x = x.repeat(1, K, 1)
                    pred = model(x) 
                    next_point = pred.data.cpu().numpy()[0,0] * Rated_Capacity
                    test_x.append(next_point)      # The test values are added to the original sequence to continue to predict the next point
                    fixed_point_list.append(next_point) # Saves the predicted value of the last point in the output sequence
                    
                    x = np.reshape(np.array(test_sequence[t:t+feature_size])/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device) 
                    x = x.repeat(1, K, 1)
                    pred = model(x) 
                    next_point = pred.data.cpu().numpy()[0,0] * Rated_Capacity
                    moving_point_list.append(next_point) # Saves the predicted value of the last point in the output sequence
                    t += 1
                    
                y_fixed_slice.append(fixed_point_list)             # Save all the predicted values
                y_moving_slice.append(moving_point_list)

                loss_list.append(loss)
                rmse = evaluation(y_test=test_data, y_predict=y_fixed_slice[-1])
                re = relative_error(y_test=test_data, y_predict=y_fixed_slice[-1], threshold=Rated_Capacity*0.7)
                #print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, re))
                
            if metric == 're':
                score = [re]
            elif metric == 'rmse':
                score = [rmse]
            else:
                score = [re, rmse]
            if (epoch + 1)%10 == 0:
                print(f'epoch:{epoch+1:2d} | loss:{loss:6.4f} | RMSE:{rmse:6.4f} | RE:{re:6.4f}')
                print(f'score: {score}')
                print(f'score_: {score_}')
            if (loss < 1e-3) and (score_[0] < score[0]): # when loss is small enough and RE is worse than before, use last score as best
                break
            score_ = score.copy()
            
        score_list.append(score_)
        fixed_result_list.append(train_data.copy() + y_fixed_slice[-1])
        moving_result_list.append(train_data.copy() + y_moving_slice[-1])
        
    return score_list, fixed_result_list, moving_result_list
