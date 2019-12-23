# -*- coding: utf-8 -*
import argparse
import torch
import random
import time
import math
from utils import *
from datasets import *
from models import *
from options import *
from torch.optim import lr_scheduler

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(all_letters,letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(all_letters,line):
    n_letters = len(all_letters)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(all_letters,letter)] = 1
    return tensor

def categoryFromOutput(selected_classes, output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return selected_classes[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair(selected_classes, all_letters, train_category_lines):
    category = randomChoice(selected_classes)
    line = randomChoice(train_category_lines[category])
    category_tensor = torch.LongTensor([selected_classes.index(category)])                    #### labels
    line_tensor = lineToTensor(all_letters, line)
    return category, line, category_tensor, line_tensor

def train(model, optimizer, criterion, category_tensor, line_tensor):
    hidden = model.initHidden().cuda()
    # cell = model.initCell().cuda()
    optimizer.zero_grad()
    # output = rnn(line_tensor, hidden, cell)
    output = model(line_tensor, hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

# Just return an output given a line
def evaluate(model, line_tensor):
    hidden = model.initHidden().cuda()
    # cell = model.initCell().cuda()
    # output = model(line_tensor.cuda(), hidden, cell)
    output = model(line_tensor.cuda(), hidden)
    return output
def predict(model, selected_classes, all_letters, line, n_predictions=1):
    output = evaluate(model, lineToTensor(all_letters, line))
    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []
    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        predictions.append([value, selected_classes[category_index]])
    return predictions

if __name__=='__main__':
    # data 
    parser = argparse.ArgumentParser(description='PyTorch NameData Example')
    parser_trained=TrainOptions(parser)
    parser_trained.initialize_namedata()
    opts = parser_trained.parser.parse_args()
    name_dataset = NameData()
    
    # Keep track of losses for plotting
    criterion = nn.NLLLoss()    
    for ii in range(opts.tasks,opts.tasks+3):    #cifar100, 50 tasks
        my_path = os.path.join("weights/",opts.model_path,str(ii))
        if not os.path.exists(os.path.join("weights/",opts.model_path)):
            os.mkdir(os.path.join("weights/",opts.model_path))
        if not os.path.exists(my_path):
            os.mkdir(my_path)
        criterion = nn.CrossEntropyLoss()
        indexes = [2*ii, 2*ii+1]
        train_category_lines, test_category_lines = name_dataset.initialize(indexes)   #### identical split

        for kk in range(100):       
            # torch.manual_seed(opts.seed) # remove this line to obtain different initialized trained models.
            all_losses,current_loss = [], 0
            start = time.time()
            print('indexes:',indexes)
            selected_classes = [name_dataset.all_categories[ind] for ind in indexes]
            rnn = RNN(name_dataset.n_letters, opts.n_hidden, 2).cuda()   ######### 

            optimizer = torch.optim.SGD(rnn.parameters(), lr=opts.lr)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opts.step1], gamma=0.1)
            
            # train:
            for epoch in range(1, opts.epochs + 1):
                category, line, category_tensor, line_tensor = randomTrainingPair(selected_classes, name_dataset.all_letters, train_category_lines)
                output, loss = train(rnn, optimizer, criterion, category_tensor.cuda(), line_tensor.cuda())
                current_loss += loss    

                # Print epoch number, loss, name and guess
                if epoch % opts.log_interval == 0:
                    guess, guess_i = categoryFromOutput(selected_classes,output)
                    correct = 'Y' if guess == category else 'N(%s)' % category
                    print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch*1.0 / opts.epochs * 100.0, timeSince(start), loss, line, guess, correct))    
                    scheduler.step() 
                # Add current loss avg to list of losses
                if epoch % opts.log_interval == 0:
                    all_losses.append(current_loss / opts.log_interval)
                    current_loss = 0    
   
                    #### test:
                    correct_num,test_num = 0, 0
                    # for i in range(name_dataset.n_categories):
                    for i in range(2):
                        one_class = selected_classes[i]
                        one_cls_list = test_category_lines[one_class]
                        test_num = test_num+len(one_cls_list)
                        for j in range(len(one_cls_list)):
                            # pres = predict(one_cls_list[j])
                            pres = predict(rnn, selected_classes, name_dataset.all_letters, one_cls_list[j])
                            if pres[0][1] == one_class:
                                correct_num = correct_num +1
                    acc = correct_num*1.0/test_num
                    print('test acc:', acc)
                    print('n_categories:', len(selected_classes))    
        
            if not (opts.not_save_model):
                print('save model ...')
                file_path = os.path.join(my_path,str(kk)+".pt")
                torch.save(rnn.state_dict(),file_path)

