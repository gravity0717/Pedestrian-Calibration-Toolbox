import numpy as np 
import matplotlib.pyplot as plt 
import logging

class UniformSampler:
    def __init__(self):
        pass 

    def run_sample(self, a, sample_step = 2):
        filter_index = [i for i in range(0, len(a), sample_step)]
        return filter_index

class SpatialSampler: 
    def __init__(self, cam_res, step: (int,int) = (30,30), DEBUG=False):
        self.cam_res = cam_res
        self.step = step
        self.LIST = [[]]
        self.DEBUG = DEBUG 
        self.size_x = None
        self.size_y = None  
    
    def set_grid(self):
        # Make a meshgrid from cam_res    
        self.size_x = int(self.cam_res[0]/self.step[0])
        self.size_y = int(self.cam_res[1]/self.step[1])
        self.x, self.y = np.meshgrid(np.arange(self.size_x), np.arange(self.size_y))        
        self.LIST = np.array([[False] * self.size_x] * self.size_y)
    
    def visualize(self):
        if self.DEBUG:
            print('*** LIST ***')
            print(f'LIST Shape:{self.LIST.shape}')
            
            import matplotlib.pyplot as plt 
            plt.scatter(self.x, self.y)
            
            # visualize occupied index
            plt.scatter(self.x[self.LIST], self.y[self.LIST], c='r')
            plt.show()
        else: logging("Error: DEBUG mode is deactivated")
        
    def is_occupied(self, a):

        ax = int(a[0]/self.step[0])
        ay = int(a[1]/self.step[1])
        
        if (0 < ax and ax <= self.size_x) and (0 < ay and ay <= self.size_y):    
            if self.LIST[ay-1][ax-1]: # [y][x] ?
                return True
            return False
        else:
            return True
    
    def make_occupied(self, a):
        if not self.is_occupied(a):
            ax = int(a[0]/self.step[0])
            ay = int(a[1]/self.step[1])
            self.LIST[ay-1][ax-1] = True  # 

    def run_sample(self, a):
        self.set_grid()
        # O(n)
        filter_mask = []
        for ai in a: 
            if not self.is_occupied(ai):
                self.make_occupied(ai)
                filter_mask.append(True)
            else:
                filter_mask.append(False)
        return np.array(filter_mask)
            
def test():
    cam_res =(20, 10)   
    ss = SpatialSampler(cam_res, (1,1), DEBUG = True)
    ss.set_grid()
    
    # 1. visualize the Grid 
    # ss.visualize()
    
    # 2. test make_occupied 
    # test_a = np.array([[20,10],[1,5],[1,3]])
    # for a in test_a:
    #     ss.make_occupied(a)  
    # ss.visualize()
    
    # 3. run sampler in real data 
    
    import matplotlib.pyplot as plt
    from lines_synthetic import gen_3d_lines_cone,gen_3d_lines_line, project_3d_pts, add_gaussian_noise
    from copy import deepcopy
    
    # Config
    line_n = 100
    line_length = 1 
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    # cam_distort = [0, 0]
    test_sampler = SpatialSampler((1920, 1080), (30, 30), DEBUG = True)
    noise_std = 5
    
    # Make data
    AB = gen_3d_lines_line(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab = add_gaussian_noise(ab, noise_std)    
    
    plt.figure(figsize=(9,6))
    a_raw = ab[::2,:]
    b_raw = ab[1::2,:]

    # Original
    plt.subplot(131) 
    plt.title("Original", fontsize = 15)
    a = deepcopy(a_raw)
    b = deepcopy(b_raw)
    for ai, bi in zip(a,b):
        plt.plot([ai[0],bi[0]],[ai[1],bi[1]], color = 'g')
    plt.xlabel(f"Num of data: {len(a)}")

    # Uniform sampling
    plt.subplot(132) 
    plt.title("Index sampling", fontsize = 15)
    a = deepcopy(a_raw)
    b = deepcopy(b_raw)
    sample_index = sample_uniform(a, 2)
    a = a[sample_index]
    b = b[sample_index]
    for ai, bi in zip(a,b):
        plt.plot([ai[0],bi[0]],[ai[1],bi[1]], color = 'g')
    plt.xlabel(f"Num of data: {len(a)}")
    
    # Spatial sampling
    plt.subplot(133)    
    plt.title("Spatial Sampling", fontsize = 15)
    a = deepcopy(a_raw)
    b = deepcopy(b_raw)
    sample_index = test_sampler.run_sample(a)
    a = a[sample_index]
    b = b[sample_index]
    for ai, bi in zip(a,b):
        plt.plot([ai[0],bi[0]],[ai[1],bi[1]], color = 'r')
    plt.xlabel(f"Num of data: {len(a)}")
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    test()  