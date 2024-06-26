import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize,ToPILImage
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DepthMapExtractor(nn.Module):
    def __init__(self, midas_model):
        super(DepthMapExtractor, self).__init__()
        self.midas = midas_model
        self.transform = Compose([
            Resize(128),  # Adjust based on your model's expected input size
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def forward(self, images):
        batch_size, num_frames, channels, height, width = images.shape
        depth_maps = torch.empty((batch_size, num_frames, height, width), dtype=torch.float32, device=images.device)
        
        for i in range(batch_size):
            for j in range(num_frames):
                # Convert the PyTorch tensor to PIL for processing
                image = ToPILImage()(images[i, j].cpu()).convert("RGB")
                image = self.transform(image).unsqueeze(0).to(images.device)
                
                # Predict depth
                with torch.no_grad():
                    depth_map = self.midas(image)
                    depth_map = torch.nn.functional.interpolate(
                        depth_map.unsqueeze(1),
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                
                depth_maps[i, j] = depth_map

        return depth_maps




def scale_invariant_log_mse(gt_depth_maps, output_depth_maps):
    """
    Calculate the Scale-Invariant Log Mean Squared Error (SI-LMSE) between ground truth and predicted depth maps.
    """
    # Ensure positive depth values
    gt_depth_maps = torch.clamp(gt_depth_maps, min=1e-6)
    output_depth_maps = torch.clamp(output_depth_maps, min=1e-6)
    
    # Compute the logarithm of depth maps
    log_gt = torch.log(gt_depth_maps)
    log_pred = torch.log(output_depth_maps)
    
    # Calculate the difference in log space
    log_diff = log_pred - log_gt
    
    # Compute the scale-invariant term
    n = torch.numel(log_gt)  # Number of elements
    scale_invariant_term = torch.sum(log_diff) / n
    
    # Calculate the Scale-Invariant Log Mean Squared Error
    si_log_mse = torch.mean((log_diff - scale_invariant_term) ** 2)
    
    return si_log_mse



class DepthMapLoss(nn.Module):
    def __init__(self, midas_model):
        super(DepthMapLoss, self).__init__()
        # Initialize the depth map extractor with the pre-loaded MiDaS model
        self.depth_extractor = DepthMapExtractor(midas_model=midas_model)

    def forward(self, data_input, model_output):
        """
        Computes a loss based on the difference between depth maps extracted from model outputs and targets.
        
        data_input: dictionary containing 'targets' as a batch of ground truth images.
        model_output: dictionary containing 'outputs' as a batch of images generated by the model.
        """
        # Extract depth maps
        gt_depth_maps = self.depth_extractor(data_input['targets'])
        output_depth_maps = self.depth_extractor(model_output['outputs'])
        #print(gt_depth_maps.shape)
        #print(output_depth_maps.shape)
        # Compute the loss between the depth maps, e.g., Mean Squared Error (MSE)

        loss = F.l1_loss(gt_depth_maps, output_depth_maps)

        
        return loss
