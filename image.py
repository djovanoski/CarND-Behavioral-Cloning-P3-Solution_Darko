import numpy as np 
import scipy
from scipy import linalg
from scipy import ndimage
import pickle
from list_iterator import ListItterator

class ImageGenerator:
    def __init__(self, horizontal_flip=False, vertical_flip=False, rotation_range=0, rescale=127.5):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.channel_axis = 3
        self.row_axis = 1
        self.col_axis = 2
        self.rescale = rescale
        self.cval = 0.
    
    def get_random_transform(self, img_shape, seed=None):
        if self.rotation_range:
            theta = np.random.uniform( -self.rotation_range,self.rotation_range)
        else:
            theta = 0
        
        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip
        
        tranform_parameters = {'flip_horizontal': flip_horizontal,
                               'flip_vertical': flip_vertical,
                               'theta': theta}
        
        
        return tranform_parameters
    
    def flip_axis(self,x,axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
    
    def apply_affine_transofrmation(self,x, 
                                    theta=0,
                                    row_axis=0, 
                                    col_axis=1, 
                                    channel_axis=2,
                                    fill_mode='nearest', 
                                    cval=0.):
        transform_matrix = None
        if theta != 0:
            theta = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
            transform_matrix = rotation_matrix
            
        if transform_matrix is not None:
            h, w = x.shape[row_axis], x.shape[col_axis]
            transform_matrix = self.transform_matrix_offset_center(transform_matrix, h, w)
            x = np.rollaxis(x, channel_axis, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]  
            channel_images = [scipy.ndimage.interpolation.affine_transform(x_channel,
                                                                           final_affine_matrix,
                                                                           final_offset,
                                                                           order=1, mode=fill_mode,
                                                                           cval=cval) for x_channel in x]
            
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
            
        return x
    
    def transform_matrix_offset_center(self,matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix
        
    
    def apply_transform(self, x, transform_parameters):
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = self.apply_affine_transofrmation(x, 
                                             transform_parameters.get('theta', 0),
                                             row_axis=img_row_axis,
                                             col_axis=img_col_axis,
                                             channel_axis=img_channel_axis,
                                             fill_mode='nearest',
                                             cval=self.cval)

        if transform_parameters.get('flip_horizontal', False):
            x = self.flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = self.flip_axis(x, img_row_axis)
            
        return x
    
    def standardize(self, x):
        x = (x - self.rescale) / self.rescale
        return x
        
    def flow_from_list(self,list_, target_size=(160,320,3), batch_size=32, shuffle=True, seed=None):
        return ListItterator(list_, self, target_size=target_size, batch_size=batch_size, shuffle=shuffle, seed=seed)