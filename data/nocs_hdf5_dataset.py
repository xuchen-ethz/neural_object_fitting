from PIL import Image
from data.base_dataset import BaseDataset, get_transform
import random
import numpy as np
import h5py
from PIL import Image
import os
class NOCSHDF5Dataset(BaseDataset):


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        input_nc = self.opt.output_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1),method=Image.BILINEAR)

        hdf5_file = h5py.File(os.path.join(opt.dataroot,opt.category+'.hdf5'),'r',swmr=True)
        self.images = hdf5_file['images']
        self.poses = hdf5_file['poses'][...]

        self.dataset_size = self.poses.shape[0]
        self.num_view = opt.n_views
        self.num_model = self.dataset_size //  self.num_view

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        model_id = random.randint(0,self.num_model-1)

        image_id = random.randint(0,self.num_view-1)
        id = model_id*self.num_view + image_id
        A_img = np.copy(self.images[id,:,:,:])
        elev,azi = np.copy(self.poses[id,:])
        if A_img.shape[2] == 4:
            A_mask = A_img[:,:,-1] == 0
            A_img[A_mask,:3] = 255
            A_img = A_img[:,:,:3]

        A = self.transform(Image.fromarray(A_img.astype(np.uint8)))
        A_pose = np.array([elev,azi,0]).astype(np.float32)

        image_id = random.randint(0,self.num_view-1)
        id = model_id*self.num_view + image_id
        B_img = np.copy(self.images[id,:,:,:])
        elev,azi = np.copy(self.poses[id,:])
        if B_img.shape[2] == 4:
            B_mask = B_img[:,:,-1] == 0
            B_img[B_mask,:3] = 255
            B_img = B_img[:,:,:3]

        B = self.transform(Image.fromarray(B_img.astype(np.uint8)))
        B_pose = np.array([elev,azi,0]).astype(np.float32)

        return {'A': A, 'A_pose': A_pose,
                'B': B, 'B_pose': B_pose,}
    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size