import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from urllib.parse import urlparse
from io import BytesIO


#PIL => python imaging library, Image module provides a class with the same name used to represent a pil image

#BytesIO => just like variables this is used to store data in bytes in an in-memory buffer

#openCV => open source computer vision library, it is a machine learning software library

#urllib.parse, urlparse => splits the url into its component pieces
                            # refer https://pymotw.com/2/urlparse/




#print("all ok")

class ColorAnalyzer:
    '''
    This class analyzes the predominant colors in an image 
    using K-Means clustering based on the RGB color paradigm.
    
    Attributes:
        url_or_path (str): The URL or local file path of the image.
        num_clusters (int): The number of clusters to identify as predominant colors.
        scaling_factor (int): The percentage by which to scale the image for preprocessing.
        image (numpy.ndarray): The loaded and preprocessed image.
        pixels (numpy.ndarray): Reshaped image data for clustering.
        image_rgb (numpy.ndarray): Resized image in RGB format.
        centroids (numpy.ndarray): Centroids (predominant colors) obtained through clustering.
        percentages (numpy.ndarray): Percentage of pixels belonging to each cluster.
        labels (numpy.ndarray): Labels indicating cluster membership for each pixel.
        sorted_colors (numpy.ndarray): Predominant colors sorted by cluster size.
        sorted_percentages (numpy.ndarray): Percentages of pixels per cluster, sorted by cluster size.
    '''
    
    def __init__(self, url_or_path, num_clusters=4, scaling_factor = 10):
        
        
        '''
        Initializes the ImageColorAnalyzer with the provided parameters.

        Args:
            url_or_path (str): The URL or local file path of the image.
            num_clusters (int, optional): The number of clusters to identify as predominant colors (default is 5).
            scaling_factor (int, optional): The percentage by which to scale the image for preprocessing (default is 10).
        '''
        
        self.url_or_path = url_or_path
        self.num_clusters = num_clusters
        self.scaling_factor = scaling_factor
        self.image = self.load_image()
        self.pixels, self.image_rgb = self.preprocess_image()
        self.centroids, self.percentages, self.labels = self.find_clusters()
        self.sorted_colors, self.sorted_percentages = self.sort_clusters_by_size()
        
        
        
    def load_image(self):
        
        '''
        Load the image into a 2D array from the local path or URL.
        
        Returns:
            numpy.ndarray: The loaded image.
        
        Raises:
            Exception: If the URL does not exist or is broken, or if the image path is invalid.
        '''
        
        #if true then take url else check for path
        
        if self.is_url():
            
            response = requests.get(self.url_or_path)
            
            #check if there was any error
            
            if response.status_code != 200:
                
                raise Exception("URL does not exist or it is broken.")
                
            #try to obtain image
            try:
                
                image = Image.open(BytesIO(response.content))
                
                #convert image from PIL to OpenCV format
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                #PIL format is rgb and openCV format is bgr so we have to convert the PIL image to bgr
            
            #raise an exception if failed
            except:
                
                raise Exception("URL may not contain an image.")
        
        #if the input is path not url
        else:
            
            #try to load image
            
            try:
                image = cv2.imread(self.url_or_path)
                
            #if there is a problem reading the image
            except:
                
                raise Exception("Invalid image path.")
            
        return image
            
        
    def is_url(self):
        
        '''
        Check if the input path is URL.
        
        Returns:
            bool: True if the path is a URL, False otherwise.
        '''
        #return true if path is url else return false
        
        return 'http' in urlparse(self.url_or_path).scheme
        
    def preprocess_image(self):
        
        '''
        Resize the image to improve processing speed.
        
        Returns:
            tuple: 
                Resized image data for clustering
                Image for plotting.
        '''
        #resize the image for performances
        
        width = int(self.image.shape[1] * self.scaling_factor/100)
        height = int(self.image.shape[0] * self.scaling_factor/100)
        
        resized_img = cv2.resize(self.image, (width,height), interpolation = cv2.INTER_AREA)
        #interpolation is the process of estimating unknown values when certain values are known
        
        #convert the image back to rgb
        image_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        #extract pixels as 2D array for clustering
        pixels = image_rgb.reshape(-1,3)
        
        #remember to try and print "image_rgb" before reshaping
        
        #print(type(image_rgb))
        #<class 'numpy.ndarray'>
        
        return pixels, image_rgb
    
    def find_clusters(self):
        
        '''
        Find predominant colors through clustering.
        
        Returns:
            tuple: 
                centroids (predominant colors)
                percentages of pixels per cluster
                labels of each point.           
        '''
        
        #instantiate clustering model
        kmeans = KMeans(n_clusters= self.num_clusters, n_init = 10)
        
        #ft the model on the image and get labels
        labels = kmeans.fit_predict(self.pixels)
        
        #get centroids
        centroids = kmeans.cluster_centers_.round(0).astype(int)
        
        #get percentage of pixels belonging to each cluster
        percentages = np.bincount(labels) / len(self.pixels)*100
        
        # return centroids, percentage of pixels per cluster and labels of each point
        
        return centroids, percentages, labels
        
    def sort_clusters_by_size(self):
        
        '''
        Sort predominant colors and percentages 
        of pixels per cluster by cluster size 
        in descending order.
        
        Returns:
            tuple: 
                Predominant colors sorted by cluster size
                Percentages of pixels per cluster sorted by cluster size.
        '''
        
        sorted_indices = np.argsort(self.percentages)[::-1]
        sorted_colors = self.centroids[sorted_indices]
        sorted_percentages = self.percentages[sorted_indices]
        
        return sorted_colors, sorted_percentages
    
        
    def plot_image(self):
        
        '''
        Plot the preprocessed image (resized).
        '''
        plt.imshow(self.image_rgb)
        plt.title("preprocessed image")
        plt.axis('off')
        plt.show()
        
        
    def plot_3d_clusters(self, width=15, height=12):
        
        '''
        Plot a 3D visualization of the clustering.
        
        Args:
            width (int, optional): Width of the plot (default is 15).
            height (int, optional): Height of the plot (default is 12).
        '''
        #prepare the figure
        
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111, projection = '3d')
        
        #plot point labels with their cluster's color
        
        for label, color in zip(np.unique(self.labels), self.centroids):
            cluster_pixels = self.pixels[self.labels == label]
            r, g, b = color
            ax.scatter(cluster_pixels[:, 0],
                       cluster_pixels[:, 1],
                       cluster_pixels[:, 2],
                       c = [[r/255, g/255, b/255]],
                       label = f'Cluster {label+1}')
            
        #display title, axis labels and legend
        
        plt.title(f'Top {(self.num_clusters)} Predominant Colors')
        plt.xlabel('r')
        plt.ylabel('g')
        plt.xticks(rotation = 45)
        plt.show()
        
        
        
        
    def plot_predominant_colors(self, width=12, height=8):
        
        '''
        Plot a bar chart of predominant colors 
        ordered by presence in the picture.
        
        Args:
            width (int, optional): Width of the plot (default is 12).
            height (int, optional): Height of the plot (default is 8).
        '''
        # Prepare color labels for the plot
        color_labels = [f'Color {i+1}' for i in range(self.num_clusters)]
        
        #prepare figure
        plt.figure(figsize = (width, height))
        
        #plot bars
        bars = plt.bar(color_labels,
                       self.sorted_percentages,
                       color = self.sorted_colors/255.0,
                       edgecolor = 'black')
        
        for bar, percentage in zip(bars, self.sorted_percentages):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f'{percentage:.2f}%',
                ha = 'center',
                va = 'bottom')
            
        plt.title(f'Top {self.num_clusters} Predominant colors')
        plt.xlabel('colors')
        plt.ylabel("Percentage of pixels")
        plt.xticks(rotation = 45)
        plt.show()
        
        
    def get_predominant_colors(self):
        
        '''
        Return a list of predominant colors.
        Each color is a JSON object with RGB code and percentage.
        '''
        
        #prepare output list
        colors_json = []
        
        #for each predominant color
        for color, percentage in zip(self.sorted_colors,
                                     self.sorted_percentages):
            
            #get rgb code
            r, g, b = color
            
            #prepare json object
            color_entry = {'color': {'R': f'{r}', 
                                     'G': f'{g}', 
                                     'B': f'{b}'}, 
                           'percentage': f'{percentage:.2f}%'}
            
            #append json object to color list
            
            colors_json.append(color_entry)
            
        return colors_json


