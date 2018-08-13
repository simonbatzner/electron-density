import matplotlib.pyplot as plt
import numpy as np

""""
Steven Torrisi,
"""

#TODO Write this
def uncertainty_plot(structure,err_thres,ran_frames=[],file_output=None):
    """
    When passed a list of frames, plot the uncertainty
    associated with each point in the simulation.

    """
    pass

    trajectory= structure.trajectory
    frames = trajectory.keys()
    uncertainties = [trajectory[frame]['uncertainty'] for frame in frames]

    mean_uncs = [np.mean(unc) for unc in uncertainties]
    # Determine vertical line points
    #train_frames=[]
    #for fr in [int(fr) for fr in frames]:
    #    if np.mean(uncertainties[frames])<=.0000001:
    #        train_frames.append(fr)


    plt.figure()

    plt.semilogy(frames,mean_uncs,marker='o',c='b',
                linestyle='dashed')

    #for x in train_frames:
    #    ax.axvline(x,linestyle='-')
    plt.title("Uncertainty by Frame")

    plt.ylim(ymin=1.0e-12)
    plt.ylim(ymax= np.max([np.max(mean_uncs)*5,err_thres*5]))

    plt.axhline(err_thres,c='r',linestyle='dashed')

    for n in ran_frames:
        plt.axvline(n,c='g',linestyle='dashed')

    plt.show()





