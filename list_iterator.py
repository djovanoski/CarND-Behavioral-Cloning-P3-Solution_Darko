from base_iterator import Iterator
import numpy as np
import multiprocessing.pool
import matplotlib.image as mpimg

class ListItterator(Iterator):
    def __init__(self, list_, image_data_generator, target_size, batch_size=32, shuffle=True, seed=None, dtype='float32'):
        super(ListItterator, self).common_init(image_data_generator, target_size)
        self.data = list_
   
        self.dtype = dtype
        self.seed = seed
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.target_size = target_size
        self.length_data = self.data[0].shape[0]
        super(ListItterator, self).__init__(self.length_data, self.batch_size, self.shuffle, self.seed)
    
    def _get_batches_of_transformed_samples(self, index_array) :
        batch_image_center = np.zeros((len(index_array),) + self.target_size,dtype=self.dtype)
        batch_speed = np.zeros((len(index_array),),dtype=self.dtype)
        batch_throttle = np.zeros((len(index_array),),dtype=self.dtype)
        batch_steering = np.zeros((len(index_array),),dtype=self.dtype)
        pool = multiprocessing.pool.ThreadPool(self.batch_size)
        all_images = [pool.apply_async(self.read_img, ('data_org/data/'+self.data[0][j],)) for i, j in enumerate(index_array)]
        for i,b in enumerate(all_images):
            batch_image_center[i] = b.get()
        for i, j in enumerate(index_array):
            batch_speed[i] = round(self.data[1][j],2)
            batch_throttle[i] = round(self.data[2][j],2)
            batch_steering[i] = round(self.data[3][j],2)
        pool.close()
        pool.join()
        return [batch_image_center, batch_speed], [batch_steering, batch_throttle]
    def read_img(self, image):
        #img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        img =mpimg.imread(image)
        params = self.image_data_generator.get_random_transform(img.shape)
        img = self.image_data_generator.apply_transform(img, params)
        img = self.image_data_generator.standardize(img)
        return img
    
    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)