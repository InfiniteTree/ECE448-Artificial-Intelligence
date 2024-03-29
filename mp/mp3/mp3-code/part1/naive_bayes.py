import numpy as np

class NaiveBayes(object):
    def __init__(self,num_class,feature_dim,num_value):
        """Initialize a naive bayes model. 

        This function will initialize prior and likelihood, where 
        prior is P(class) with a dimension of (# of class,)
            that estimates the empirical frequencies of different classes in the training set.
        likelihood is P(F_i = f | class) with a dimension of 
            (# of features/pixels per image, # of possible values per pixel, # of class),
            that computes the probability of every pixel location i being value f for every class label.  

        Args:
            num_class(int): number of classes to classify
            feature_dim(int): feature dimension for each example 
            num_value(int): number of possible values for each pixel 
        """

        self.num_value = num_value
        self.num_class = num_class
        self.feature_dim = feature_dim

        self.prior = np.zeros((num_class))
        self.likelihood = np.zeros((feature_dim,num_value,num_class))


    def train(self,train_set,train_label):
        """ Train naive bayes model (self.prior and self.likelihood) with training dataset. 
            self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
            self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of 
                (# of features/pixels per image, # of possible values per pixel, # of class).
            You should apply Laplace smoothing to compute the likelihood. 

        Args:
            train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
            train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
        """

        # YOUR CODE HERE
        for idx in range(len(train_label)):
            cla=train_label[idx]
            self.prior[cla]+=1
            for pix in range(self.feature_dim):
                self.likelihood[pix,int(train_set[idx][pix]),cla]+=1

        # Laplace smoothing
        k=1
        for cla in range(self.num_class):
            tot=self.prior[cla]+self.num_value*k
            self.likelihood[:,:,cla]+=k
            self.likelihood[:,:,cla]/=tot

        self.prior/=len(train_label)


        #avoid underflow
        self.likelihood=np.log(self.likelihood)
        self.prior=np.log(self.prior)
        print("Trained")

    def test(self,test_set,test_label):
        """ Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
            by performing maximum a posteriori (MAP) classification.  
            The accuracy is computed as the average of correctness 
            by comparing between predicted label and true label. 

        Args:
            test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
            test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

        Returns:
            accuracy(float): average accuracy value  
            pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
        """    

        # YOUR CODE HERE

        accuracy = 0
        pred_label = np.zeros((len(test_set)))
        min_fig=np.zeros((10,2))
        max_fig=np.zeros((10,2))

        for idx in range(len(test_label)):
            ev=np.zeros(self.num_class)
            for cla in range(self.num_class):
                ev[cla] = self.prior[cla]
                for pix in range(self.feature_dim):
                    ev[cla]+=self.likelihood[pix,int(test_set[idx][pix]),cla] #since we use log above.
            label=np.argmax(ev)
            pred_label[idx]=label
            
            if(label==test_label[idx]):
                if(min_fig[label][1]==0 or min_fig[label][1]>ev[label]):
                    min_fig[label][0]=idx
                    min_fig[label][1]=ev[label]
                if(max_fig[label][1]==0 or max_fig[label][1]<ev[label]):
                    max_fig[label][0]=idx
                    max_fig[label][1]=ev[label]
                accuracy+=1
        accuracy/=len(test_label)
        print("Tested")
        # return accuracy, pred_label, min_fig, max_fig
        return accuracy, pred_label


    def save_model(self, prior, likelihood):
        """ Save the trained model parameters 
        """    

        np.save(prior, self.prior)
        np.save(likelihood, self.likelihood)

    def load_model(self, prior, likelihood):
        """ Load the trained model parameters 
        """ 

        self.prior = np.load(prior)
        self.likelihood = np.load(likelihood)

    def intensity_feature_likelihoods(self, likelihood):
        
        """
        Get the feature likelihoods for high intensity pixels for each of the classes,
            by sum the probabilities of the top 128 intensities at each pixel location,
            sum k<-128:255 P(F_i = k | c).
            This helps generate visualization of trained likelihood images. 
        
        Args:
            likelihood(numpy.ndarray): likelihood (in log) with a dimension of
                (# of features/pixels per image, # of possible values per pixel, # of class)
        Returns:
            feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
                (# of features/pixels per image, # of class)
        """
        # YOUR CODE HERE
        feature_likelihoods = np.sum(np.exp(likelihood[:,128:,:]),axis=1)   #since we use log above
        return feature_likelihoods
        
    
    
