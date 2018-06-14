# DeepClas4Bio

DeepClas4Bio is a project that aims to facilitate the interoperability of bioimaging tools with deep learning frameworks.
In particular, we have developed an extensible API that provides a common access point for classification models of
several deep learning frameworks. This project groups the main deep learning frameworks, namely, Keras, Caffe, 
DeepLearning4J, MxNet and PyTorch.

Using DeepClas4Bio, the users could work with the main deep learning frameworks easily and transparently for their. 
Even the users can use the pretrained models included in the API for classify an image or they can train their own 
models and load it in the API to use it in a simple way.

## System requirements

DeepClas4Bio requieres Python 3.* and numpy.
Also it needs the installation of the framework that the user want to use (Keras, Caffe, MxNet, PyTorch or DL4J).

## Including new frameworks in the API
To include new frameworks in the API you have to complete these three steps:

 1. Create the corresponding model class i.e. *FrameworkModel* class that inherits from Model. This is an empty 
 class to relate a predictor an a model.
 2. Create the corresponding predictor class called *FrameworkPredictor* that inherits from Predictor. 
 In this class the predict method of this framework must be implemented.
 3. Create a file with the functions of the models. This class called *FrameworkFunctions* groups the methods to
 load a model of this framework, preprocess the input and postprocess the output. Also, in this class the list of
 models available in this framework is collected.
 
 All of these files must be placed in python package called *Framework*.

## Including new model in the API
To include new models in the API you have to complete these three steps:

 1. Save the structure's file in the path *Framework/Classification/model/ModelName* and the weigths' file in 
 the path *Framework/Classification/weights/ModelName*.
 2. Add to corresponding *FrameworkFunctions* file thre methods. A method called *ModelNameFrameworkload* to load the 
 model (for example vgg16kerasload). The second method is responsible of preprocess the input image and its called
 *ModelNameFrameworkpreprocess* (for example vgg16keraspreprocess). And the last method is to postprocess the result,
 this method must follow the following nomenclature *ModelNameFrameworkpostprocess* (for example vgg16keraspostprocess).
 3. Finally, the model (its name) must be added to the list of models available in the framework. This list is in the
 corresponding *FrameworkFunctions* file.
 
## Using the API
This API has been developed to connect deep learning techniques with bioimage programs like ImageJ, ImagePy or Icy
for example. But it can be used to other purposes. In the following lines we will explain the different ways to use
this API.

DeepClas4Bio has three different methods:
 1. List available frameworks.
 2. List available models in a framework.
 3. Classify an image.
 
 These methods could be used from the command line, from a web API or better from bioimage tools.
### Command line
Using the command line you have to execute the following command in the root folder of the project.


- List available frameworks.

````
    python listFrameworks.py
````

- List available models in a framework.

````
    python listModels.py - f NameOfTheFramework
````

- Classify an image.

````
    python predict.py -i pathToTheImage - f NameOfTheFramework -m NameOfTheModel
````

An example of use of this API could be found in the following Collab Notebook [Using DeepClas4Bio API](https://drive.google.com/open?id=1paYEOVU6SuJiZHbFAJCKzetTZXo28mbY)

### Web API
This API could be used from a web API as presented in the server file of this project. In this file you can see
how to create a web service with Flask and Redis to classify image with a concrete model.


### Bioimage tools
This is the better way to use the API. We have connect DeepClas4Bio with three different bioimage tools, namely, 
ImageJ, ImagePy and Icy. THe source code of these plugins could be found:
- [ImageJ plugin](https://github.com/adines/DeepClas4BioIJ). This plugin connects DeepClas4Bio with ImageJ.
- [ImagePy plugin](https://github.com/adines/DeepClas4BioImagePy). This plugin connects DeepClas4Bio with ImagePy.
- [Icy plugin](https://github.com/adines/DeepClas4BioIcy). This plugin connects DeepClas4Bio with Icy.
- [ImageJ model comparator](https://github.com/adines/DeepClas4BioIJComparator). This plugin allow to compare different deep models in ImageJ.

Also we have created an ImageJ plugin metagenerator to create plugins easily. The source code of this plugins could
be found [here](https://github.com/adines/DeepClas4BioIJMetagenerator).
