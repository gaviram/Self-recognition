import numpy as np
import torch

def MSE(target, predicted):
  tar1 = target[:,1,:,:].cpu().detach().numpy()
  out1 = predicted[:,1,:,:].cpu().detach().numpy()
  return np.mean((tar1 - out1)**2)
  
def dice_score(new_targets, new_inputs,smooth=1, th = 0.1):
      inputs = new_inputs.clone().cpu().detach()
      targets = new_targets.clone().cpu().detach()
      inputs[:,1,:,:] = inputs[:,1,:,:]>(inputs[:,1,:,:]).mean()
      # inputs = torch.sigmoid(inputs)[1]
      targets[:,1,:,:] = targets[:,1,:,:]>targets[:,1,:,:].mean()
      inputs[:,0,:,:] = inputs[:,0,:,:]<inputs[:,0,:,:].mean()#.cpu().detach().numpy()
      targets[:,0,:,:] = targets[:,0,:,:]<targets[:,0,:,:].mean()#.cpu().detach().numpy()
      
      chan_dim=1
      obs_dim=0
      xy_dim = [2, 3]
      batch_size = inputs.shape[obs_dim]
      w = 1/(targets.sum(dim=xy_dim)**2+1e-10)
      intersection = (inputs*targets).sum(dim=xy_dim)
      union = (inputs**2 + targets**2).sum(dim=xy_dim)
      numer = 2 * (w*intersection).sum(dim=chan_dim)
      denom = (w*union).sum(dim=chan_dim)

      dice = ((numer) / (denom + 1e-10)).sum()/batch_size
      return float(dice)
# def dice_score(inputs, targets, smooth=1, th = 0.1):
#       inputs[:,1,:,:] = inputs[:,1,:,:]>th#.cpu().detach().numpy()
#       # inputs = torch.sigmoid(inputs)[1]
#       targets[:,1,:,:] = targets[:,1,:,:]>th#.cpu().detach().numpy()
#       inputs[:,0,:,:] = inputs[:,0,:,:]<th#.cpu().detach().numpy()
#       targets[:,0,:,:] = targets[:,0,:,:]<th#.cpu().detach().numpy()
#       chan_dim=1
#       obs_dim=0
#       xy_dim = [2, 3]
#       batch_size = inputs.shape[obs_dim]
#       w = 1/(targets.sum(dim=xy_dim)**2+1e-10)
#       intersection = (inputs*targets).sum(dim=xy_dim)
#       union = (inputs**2 + targets**2).sum(dim=xy_dim)
#       numer = 2 * (w*intersection).sum(dim=chan_dim)
#       denom = (w*union).sum(dim=chan_dim)

#       dice = ((numer) / (denom + 1e-10)).sum()/batch_size
#       return float(dice)

# def dice_score_old(pred, targs):
#     pred = (pred>0).float()
#     return 2. * (pred*targs).sum() / (pred+targs).sum()

# def dice_score_old(inputs, targets, smooth=1):
#       inputs = inputs[1]
#       # inputs = torch.sigmoid(inputs)[1]
#       targets = targets[1]
#       # xy_dim = [2,3]
#       chan_dim=1p
#       obs_dim=0
#       batch_size = inputs.shape[obs_dim]
#       w = 1/(targets.sum()**2+1e-10)
#       # flatten label and prediction tensors
#       intersection = (inputs*targets).sum()
#       union = (inputs**2 + targets**2).sum()
#       numer = 2 * (w*intersection).sum()
#       denom = (w*union).sum()

#       dice = ((numer) / (denom + 1e-10)).sum()/batch_size
#       return dice

def meanIOU(target, predicted, th = 0.05):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return

    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return

    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy()[1]
        target_arr[target_arr > np.mean(target_arr)] = 1
        target_arr[target_arr < np.mean(target_arr)] = 0
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy()[1]
        predicted_arr[predicted_arr > np.mean(predicted_arr)] = 1
        predicted_arr[predicted_arr < np.mean(predicted_arr)] = 0
        #target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)#[1]#
        #predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)#[1]#.argmin(0)

        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else:
            iou_score = intersection / union
        iousum += iou_score

    miou = iousum / target.shape[0]
    return miou


def pixelAcc(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return

    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return

    accsum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy()[1]
        target_arr[target_arr > np.mean(target_arr)] = 1
        target_arr[target_arr < np.mean(target_arr)] = 0
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy()[1]
        predicted_arr[predicted_arr > np.mean(predicted_arr)] = 1
        predicted_arr[predicted_arr < np.mean(predicted_arr)] = 0
        #target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        #predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)

        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a * b
        accsum += same / total
        #print(same / total)
    pixelAccuracy = accsum / target.shape[0]
    return pixelAccuracy

# def pixelAcc(target, predicted):    
#     if target.shape != predicted.shape:
#         print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
#         return
        
#     if target.dim() != 4:
#         print("target has dim", target.dim(), ", Must be 4.")
#         return
    
#     accsum=0
#     for i in range(target.shape[0]):
#         target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
#         predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        
#         same = (target_arr == predicted_arr).sum()
#         a, b = target_arr.shape
#         total = a*b
#         accsum += same/total
    
#     pixelAccuracy = accsum/target.shape[0]        
#     return pixelAccuracy

# def meanIOU(target, predicted):
#     if target.shape != predicted.shape:
#         print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
#         return
        
#     if target.dim() != 4:
#         print("target has dim", target.dim(), ", Must be 4.")
#         return
    
#     iousum = 0
#     for i in range(target.shape[0]):
#         target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmin(0)
#         predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmin(0)
        
#         intersection = np.logical_and(target_arr, predicted_arr).sum()
#         union = np.logical_or(target_arr, predicted_arr).sum()
#         if union == 0:
#             iou_score = 0
#         else :
#             iou_score = intersection / union
#         iousum +=iou_score
        
#     miou = iousum/target.shape[0]
#     return miou