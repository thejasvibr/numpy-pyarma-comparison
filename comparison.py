"""
Comparing the speed of PyArmadillo and NumPy implementations

"""
import pyarma as pya
import numpy as np 
matmul = np.matmul

def numpy_spwa_locn(array_geometry, d, **kwargs):
    '''
    '''
    nmics = array_geometry.shape[0]

    c = kwargs.get('c', 340.0) # m/s
    # check that the 1st mic is origin - else set it to 0,0,0
    if not np.array_equal(array_geometry[0,:], np.array([0,0,0])):
        mic1_notorigin = True
        mic1 = array_geometry[0,:]
        array_geometry = array_geometry - mic1
    else:
        mic1_notorigin = False

    # the receiver matrix- excluding the first channel.
    R = array_geometry[1:,:]
    tau = d.copy()/c # to keep symbol conventions the same
    
    R_inv = np.linalg.pinv(R)
    
    Nrec_minus1 = R.shape[0]
    b = np.zeros(Nrec_minus1)
    f = np.zeros(Nrec_minus1)
    g = np.zeros(Nrec_minus1)
    #print(R, tau)
    for i in range(Nrec_minus1):
        b[i] = np.linalg.norm(R[i,:])**2 - (c*tau[i])**2
        f[i] = (c**2)*tau[i]
        g[i] = 0.5*(c**2-c**2)
    
    a1 = matmul(matmul(R_inv, b).T, matmul(R_inv,b))
    a2 = matmul(matmul(R_inv, b).T, matmul(R_inv,f))
    a3 = matmul(matmul(R_inv, f).T, matmul(R_inv,f))
    
    # quadratic equation is ax^2 + bx + c = 0
    # solution is x = (-b +/- sqrt(b^2 - 4ac))/2a
    # replace 
    
    a_quad = a3 - c**2
    b_quad = -a2
    c_quad = a1/4.0
    
    t_solution1 = (-b_quad + np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)
    t_solution2 = (-b_quad - np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)
    t1 = (t_solution1 , t_solution2)

    s = [matmul(R_inv,b*0.5) - matmul(R_inv,f)*t1[0],
         matmul(R_inv,b*0.5) - matmul(R_inv,f)*t1[1]]

    if mic1_notorigin:
        for each in s:
            each += mic1

    return s

   
   
def pyarma_spwa_locn(array_geometry, d, **kwargs):
    '''
    '''
    nmics = array_geometry.n_rows

    c = kwargs.get('c', 340.0) # m/s
    # check that the 1st mic is origin - else set it to 0,0,0
    if pya.sum(array_geometry[0,:]== pya.mat([0,0,0]))!=3:
        mic1_notorigin = True
        mic1 = array_geometry[0,:]
        array_geometry = array_geometry - mic1
    else:
        mic1_notorigin = False

    # the receiver matrix- excluding the first channel.
    R = array_geometry[1:,:]
    tau = d/c # to keep symbol conventions the same

    R_inv = pya.pinv(R)
    
    Nrec_minus1 = R.n_rows
    b = pya.zeros(Nrec_minus1)
    f = pya.zeros(Nrec_minus1)
    g = pya.zeros(Nrec_minus1)
    #print(R, tau)
    for i in range(Nrec_minus1):
        b[i] = pya.norm(R[i,:])**2 - (c*tau[i])**2
        f[i] = (c**2)*tau[i]
        g[i] = 0.5*(c**2-c**2)
    
    
    a1 = pya.trans(R_inv*b)*(R_inv*b)
    a2 = pya.trans(R_inv*b)*(R_inv*f)
    a3 = pya.trans(R_inv*f)*(R_inv*f)

    # quadratic equation is ax^2 + bx + c = 0
    # solution is x = (-b +/- sqrt(b^2 - 4ac))/2a
    # replace 
    
    a_quad = a3 - c**2  
    b_quad = -a2
    c_quad = a1/4.0
    
    t_solution1 = (-b_quad + pya.sqrt(pya.pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad)
    t_solution2 = (-b_quad - pya.sqrt(pya.pow(b_quad,2) - 4*a_quad*c_quad))/(2*a_quad)
    t1 = (t_solution1 , t_solution2)

    s = [R_inv*b*0.5 - R_inv*f*t1[0],
         R_inv*b*0.5 - R_inv*f*t1[1]]

    if mic1_notorigin:
        for i, _ in enumerate(s):
            s[i] += pya.trans(mic1)
    return s

if __name__ == "__main__":
    import pydatemm
    from pydatemm.simdata import simulate_1source_and1reflector_general
    #from simdata import simulate_1source_and1reflector_general, simulate_1source_and_1reflector_3dtristar
    np.random.seed(825)
    audio, distmat, array_geom, (source,ref)= simulate_1source_and1reflector_general(**{'nmics':5})
    
    d = np.array([each-distmat[0,0] for each in distmat[0,1:]])
    #print('hello', source)
    #print('arraygeom:', array_geom)
    array_geom_pya = pya.mat(array_geom.tolist())
    d_pya = pya.mat(d.tolist())
    #%%
    import time
    start = time.perf_counter_ns()
    for i in range(1000):
        source_pos = pyarma_spwa_locn(array_geom_pya, d_pya, c=340)
    print(f'time taken: {(time.perf_counter_ns()-start)/10**12}:pya')
    #print(f'source positions: {source_pos}')
    #%%
    start = time.perf_counter_ns()
    for i in range(1000):
        source_pos = numpy_spwa_locn(array_geom, d, c=340)
    print(f'time taken: {(time.perf_counter_ns()-start)/10**12}: numpy')
    
    #%%
    %load_ext line_profiler
    %lprun -f numpy_spwa_locn numpy_spwa_locn(array_geom, d, c=340)
    %lprun -f pyarma_spwa_locn pyarma_spwa_locn(array_geom_pya, d_pya, c=340)
