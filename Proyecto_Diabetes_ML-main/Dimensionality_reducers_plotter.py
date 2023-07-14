from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GraphGenerator:
    "Class that receives trained pca or tsne models and the data, and returns graphics related to each of these models." 
    def __init__(self, data, pca=None,tsne = None):
        """ Constructor.

        Args:
            data (list or array): dataset to transform.

            pca (object): [trained pca model.] Defaults to None.
            
            tsne (object): [trained and transformed tsne model]. Defaults to None.
        """
        if pca is not None:
            self.pca = pca
            self.X_pca =self.pca.transform(data)
        
        if tsne is not None:
            self.tsne = tsne

    def plot_cumulative_explained_variance(self, figsize = (7,5)):
        """Calculates the explained variance of each CP and returns an array with these variances and the graph of cumulative variance vs. number of CPs. 
        
        Args: 
             figsize (tuple, optional): [size of the generated figure.]. Defaults to (10,8). 

        Outputs:
            Figure with two subplots: the first one with explained variance vs the number of components, and the second with explained variance by component.  
        """
        expl = self.pca.explained_variance_ratio_
        print(expl)

        fig, (axs1,axs2) = plt.subplots(1,2, figsize=figsize)
        fig.set_figwidth(15)
        plt.subplots_adjust(wspace = 0.4)
        fig.patch.set_facecolor('white')
        axs1.plot(np.cumsum(expl))
        axs1.set_xlabel('Number of components')
        axs1.set_ylabel('Cumulative explained variance')
        axs2.bar(range(self.X_pca.shape[1]), self.pca.explained_variance_ratio_)
        axs2.set_xlabel('Number of components')
        axs2.set_ylabel('Explained variance')
        plt.show()
        
    def __slice(self, columns = (0,1)):
        """Receives a tuple with the numbers of the columns to be compared, and returns a tuple with the value
        
        of the first column, the next of the second, and the distance between the numbers (step).
        
        Args:
            Columns (tuple): columns to be compared.
            
        Outputs: 
            Tuple to generate the required slices.
        
        """
        if columns[0] >= columns[1]:
            raise ValueError("Columns numbers must be different and incremental.")            
        
        if columns[1] == columns[0] + 1:
            step = 1 
        else:
            step = columns[1] - columns[0]
            
        return (columns[0],columns[1]+1,step)
        
    def biplot(self, labels, columns =(0,1), labl={0:'Negative',1:'Positive'}, features=None, figsize = (7,5)):
        """Generates biplot, witch plots  the data within new axes (usually the first and the 
        
        second components that are believed to explained most of the data variance) as well as loadings 
        
        of each variable on these components. 
        
        Args:
            labels (numeric list or array): dataset output labels.
                
            columns (tuple, optional): cpa's colums to be ploted. Defaults to (0,1,2).  
            
            features (numeric list or array): dateset features.
            
            labl (dict, optional): labels and their interpretation. Defaults to {0:'Negative',1:'Positive'}..
                      
            figsize (tuple, optional): size of the generated figure.. Defaults to (10,8).
                        
        Outputs:
            The biplot plot.  
        """
        
        slice = self.__slice(columns)
        coeff = np.transpose(self.pca.components_[slice[0]:slice[1]:slice[2]]) 
        
        xs = self.X_pca[:,columns[0]]
        ys = self.X_pca[:,columns[1]]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        cdict={0:'red',1:'green'}
        marker={0:'*',1:'o'}
        alpha={0:.3, 1:.5}

        fig, ax = plt.subplots(figsize=figsize)

        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter((xs*scalex)[ix],(ys*scaley)[ix],c=cdict[l],label=labl[l],s=40,marker=marker[l],alpha=alpha[l])
            plt.legend()
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'black',alpha = 0.5)
            if features is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'r', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, features[i], color = 'black',size = "large",  ha = 'center', va = 'center')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        ax.set_xlabel("PC{}".format(columns[0]+1))
        ax.set_ylabel("PC{}".format(columns[1]+1))
        plt.show()
    
    def triplot(self, labels, columns = (0,1,2), labl={0:'Negative',1:'Positive'}, figsize = (10,8)):
        """Generates a triplot witch plots the data within new axes (usually the first three 
        
        components).

        Args:
            labels (numeric list or array): dataset output labels.
            columns (tuple, optional): cpa's colums to be ploted. Defaults to (0,1,2).
            labl (dict, optional): labels and their categorical interpretation. Defaults to {0:'Negative',1:'Positive'}.
            figsize (tuple, optional): size of the generated figure. Defaults to (10,8).
            
        Output:
            3D scatter plot.
        """
        
        xs = self.X_pca[:,0]
        ys = self.X_pca[:,1]
        zs = self.X_pca[:,2]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        scalez = 1.0/(zs.max() - zs.min())
        
        cdict={0:'red',1:'green'}
        marker={0:'*',1:'o'}
        alpha={0:.3, 1:.5}
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        
        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter((xs*scalex)[ix],(ys*scaley)[ix],(zs*scalez)[ix], c=cdict[l],label=labl[l],s=40,marker=marker[l],alpha=alpha[l])
        
        ax.set_xlabel("PC{}".format(columns[0]+1))
        ax.set_ylabel("PC{}".format(columns[1]+1))
        ax.set_zlabel("PC{}".format(columns[2]+1))
        plt.legend()
        plt.show()
        
    def tsne_plotter(self,labels):
        """Generates tsne plot.

        Args:
            labels (numeric list or array): dataset output labels.
        """
        xs = self.tsne[:,0]
        ys = self.tsne[:,1]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        labl={0:"Negativo", 1:"Positivo"}
        cdict={0:'red',1:'green'}
        marker={0:'*',1:'o'}
        alpha={0:.3, 1:.5}

        fig, ax = plt.subplots(figsize=(7,5))

        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter((xs*scalex)[ix],(ys*scaley)[ix],c=cdict[l],label=labl[l],s=40,marker=marker[l],alpha=alpha[l])
            plt.legend()

        
    
 
