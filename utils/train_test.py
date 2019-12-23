import time
import torch
import numpy as np
from metrics import AverageMeter,accuracy
from utils.write_data import WriteData
from sklearn.metrics import average_precision_score


def evaluate(query_feats, query_targets, gal_feats, gallery_targets):
    # compute scores
    all_scores =  np.dot(query_feats.cpu().numpy(),gal_feats.cpu().numpy().transpose(1,0))   
    # compute mAP
    query_targets = np.array(query_targets)
    gallery_targets = np.array(gallery_targets)
    mAP = 0
    cmc = np.zeros((len(gallery_targets),1))
    for i in range(len(query_targets)):
        ranks = np.zeros((len(gallery_targets),1))
        query = query_targets[i]
        # print(type(query),type(gallery_targets))
        flags = (query == gallery_targets) 
        scores = all_scores[i,:]
        ap = average_precision_score(flags, scores)  ######
        mAP = mAP + ap
        index = np.argsort(-scores)

        correct_ones = np.argwhere(query==gallery_targets[index])
        ranks[correct_ones[0,0]:] = 1     ###
        cmc = cmc+ ranks
    
    mAP = mAP/len(query_targets)
    cmc = cmc/len(query_targets)

    return cmc, mAP


class TrainTest():
    def __init__(self, model, criterion, optimizer):
        self.model=model
        self.criterion=criterion
        self.optimizer=optimizer

    def train(self, trainloader, epoch, opts, log_file):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        self.model.train()
        end = time.time()
        for i, (inputs, target) in enumerate(trainloader):
            data_time.update(time.time() - end)
            target = target.cuda()
            output = self.model(inputs)
            loss = self.criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), target.size(0))
            top1.update(prec.item(), target.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % opts.log_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.4f}% ({top1.avg:.4f}%)'.format(
                       epoch, i, len(trainloader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1))
        files = WriteData(log_file)
        files.write_data_txt('train: {top1.avg:.4f}%'.format(top1=top1)+'    {loss.avg:.4f}'.format(loss=losses))        
    def test_cls(self, val_loader, opts, log_file):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        self.model.eval()
        end = time.time()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(val_loader):
                # inputs, target = inputs.cuda(), target.cuda()
                target = target.cuda()
                output = self.model(inputs)
                loss = self.criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), target.size(0))
                top1.update(prec.item(), target.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                if i % opts.log_interval == 0:  
                    print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.4f}% ({top1.avg:.4f}%)'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))
        print(' * Prec {top1.avg:.4f}% '.format(top1=top1))
        files = WriteData(log_file)
        files.write_data_txt('test: {top1.avg:.4f}% '.format(top1=top1))
        return top1.avg
    def test_cls_topk(self, val_loader, opts, log_file):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        top10 = AverageMeter()
        self.model.eval()
        end = time.time()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(val_loader):
                target = target.cuda()
                output = self.model(inputs)
                loss = self.criterion(output, target)
                prec = accuracy(output, target,(1,5,10))
                losses.update(loss.item(), target.size(0))
                top1.update(prec[0].item(), target.size(0))
                top5.update(prec[1].item(), target.size(0))
                top10.update(prec[2].item(), target.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                if i % opts.log_interval == 0:  
                    print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.4f}% ({top1.avg:.4f}%)'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))
        print(' * Prec {top1.avg:.4f}% '.format(top1=top1))
        print(' * Prec {top5.avg:.4f}% '.format(top5=top5))
        print(' * Prec {top10.avg:.4f}% '.format(top10=top10))
        files = WriteData(log_file)
        files.write_data_txt('test: {top1.avg:.4f}% '.format(top1=top1))
        files.write_data_txt('test: {top5.avg:.4f}% '.format(top5=top5))
        files.write_data_txt('test: {top10.avg:.4f}% '.format(top10=top10))
        return top1.avg
    def test_retr(self, query_loader, gal_loader, opts, log_file):                              #####
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        self.model.eval()
        end = time.time()

        count = 0
        query_feats = torch.FloatTensor().cuda()   # train_set, train_targets, query_set, query_targets, gallery_set, gallery_targets
        gal_feats = torch.FloatTensor().cuda()        
        query_targets, gallery_targets = [], []
        eps = 1e-8

        with torch.no_grad():
            for i, (inputs, target) in enumerate(query_loader):
                output = self.model(inputs)
                query_feats = torch.cat((query_feats,output),0)
                query_targets.append(target.item())


            for i, (inputs, target) in enumerate(gal_loader):
                output = self.model(inputs)
                gal_feats = torch.cat((gal_feats,output),0)
                gallery_targets.append(target.item())

            fnorm = torch.norm(query_feats, p=2, dim=1, keepdim=True)     
            query_feats = query_feats.div(fnorm.expand_as(query_feats)+eps)

            fnorm = torch.norm(gal_feats, p=2, dim=1, keepdim=True)        
            gal_feats = gal_feats.div(fnorm.expand_as(gal_feats)+eps)

        cmc, mAP = evaluate(query_feats, query_targets, gal_feats, gallery_targets)
        files = WriteData(log_file)
        files.write_data_txt('test: {cmc:.4}%, {mAP:.4}%'.format(cmc=cmc, mAP= mAP))
        print('cmc rank 1,5,10, mAP:',cmc[0], cmc[4],cmc[9],mAP)
        return cmc, mAP  

  