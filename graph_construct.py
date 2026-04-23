import torch
import torch.nn.functional as F


def genegene(expr_mat): 
    expr_tensor = torch.tensor(expr_mat, dtype=torch.float)
    normed_genes = F.normalize(expr_tensor, p=2, dim=0)   #normalize all gene columns
    GG_tensor = torch.matmul(normed_genes.T, normed_genes)
    return GG_tensor
    
def cellgene(expr_mat, n_bins):
    CG_tensor = torch.tensor(expr_mat, dtype=torch.float) / n_bins
    return CG_tensor

def cellcell(expr_mat, connection_type="cosim"):
    expr_tensor = torch.tensor(expr_mat, dtype=torch.float)
    normed_cells = F.normalize(expr_tensor, p=2, dim=1)   #normalize all cells individually
    CC_tensor = torch.matmul(normed_cells, normed_cells.T)
  
    
    if connection_type=="cosim":
           return CC_tensor
    
    elif connection_type=="order5":
            # Number of cells
            num_cells = CC_tensor.shape[0]

            # Percentage of neighbors to connect (5%)
            percent_neighbors = 0.05
            num_neighbors = int(num_cells * percent_neighbors)

            # Create a mask to zero out the connections that are not in the top 5% of neighbors
            mask = torch.zeros_like(CC_tensor)

            for i in range(num_cells):
                # Get the similarities for the i-th cell
                similarities = CC_tensor[i]
                
                # Find the indices of the top 5% neighbors (excluding the cell itself)
                topk_indices = torch.topk(similarities, num_neighbors + 1).indices  # +1 to include the cell itself
                topk_indices = topk_indices[topk_indices != i][:num_neighbors]  # Exclude the cell itself
                
                # Set the mask for the top 5% neighbors
                mask[i, topk_indices] = 1
            # Apply the mask to the adjacency matrix
            CC_tensor_top5 = CC_tensor * mask

            return CC_tensor_top5
    elif connection_type=="sum5":
            # Number of cells
            num_cells = CC_tensor.shape[0]

            # Percentage of neighbors to connect (5%)
            percent_neighbors= 0.05

            for i in range(num_cells):
                similarities = CC_tensor[i]
                similarity_total= torch.sum(similarities)
                percentile=similarity_total*percent_neighbors
                CC_tensor[i][CC_tensor[i]<percentile]=0

            return CC_tensor
             

