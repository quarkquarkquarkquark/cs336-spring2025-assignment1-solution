import torch
import numpy as np
import numpy.typing as npt
def process_1d_array(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    mx_id=len(dataset)-context_length
    # print("!!!",mx_id,batch_size)
    st_indices = np.random.choice(mx_id, size=batch_size, replace=False)
    ds=np.array([dataset[i:i+context_length] for i in st_indices])
    ds_t=np.array([dataset[i+1:i+context_length+1] for i in st_indices])
    ds=torch.tensor(ds).to(device)
    ds_t=torch.tensor(ds_t).to(device)

    return ds,ds_t
    