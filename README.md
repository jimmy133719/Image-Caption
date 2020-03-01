# Image-Caption
## Objective
Given an image, the model would generate a sequence of words that briefly describe the image.


## How to execute the code with your own images ?

name your image “image.jpeg” and place it to the path such as “path1/image.jpeg”. 

The pre-trained model weight and word_map json file is stored in the following drive link:  
model weight : https://drive.google.com/open?id=15PHcsNqUQjxyXFpG28E2u4kxfEAIujgy   
word map :  https://drive.google.com/open?id=1ASrqOmSGPacLEXhNEoWl9-_ZVMdgy5Ui   

Place the model weight to the path like ‘’path2/’’ and the word_map to “path3/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json” 

Then execute the following command:  
python caption.py --img='path1/image.jpeg'--model='path2/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'--word_map='path3/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=5v


## Execution steps : 
1. download dataset by clicking following link
train dataset : http://images.cocodataset.org/zips/train2014.zip
validation dataset: http://images.cocodataset.org/zips/val2014.zip 
annotation dataset: http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip 
1. run create_input_files.py to prepare model’s input
2. run train.py to train model
3. run eval.py to evaluate model
4. run caption.py to inference captions for images


#### model.py: 
The model.py defines the model used to Image Captioning. There are three components of model.py.

# Encoder
See Encoder in model.py

For implementation, we use Convolutional Neural Networks to encode our image.
Given an original image with three color channels as the input, the encoder encodes the image into smaller image with more channels.

Our goal here is to extract the important features of the input image, so we remove the linear and pooling layers of the pre-trained model. After encoding, the smaller encoded image then represents the features of the original input image.

As for the pre-trained model, after testing out quite a few different CNN models, we chose to use ResNet101, which yields the best result.

# Decoder : 
See DecoderWithAttention in model.py

After the encoder, and before decoder, it would create the initial hidden state from encoded image to feed the decoder. The decoder which is a LSTM is used to generate a sequence of words from the hidden state which could be create from using weighted average across all pixels. And this technique of weighted average is known as Attention network. In addition, each predicted word and the weighted average of the encoded image are used to generate the next word.

# Attention : 
See Attention in model.py

The Attention network would use the encoded image and previous hidden state to generate weights for each pixel.


#### create_input_files.py
Before training, run the create_input_files.py to preprocess the image dataset.


#### train.py :
All the hyperparameters are defined in the train.py and the functions written in the models.py will be imported here as well. There will be a train and validate function. The former one does the training model and the latter one does the validation to compute the bleu-4 index.


#### eval.py :
We can evaluate trained models by executing eval.py. Validation data is fed to models, and this code will calculate bleu-4 index to show models’ performance. 


#### caption.py :
caption.py is used to inference captions and visualize salient region with respect to each word in a caption.

caption_image_beam_search reads an image, encodes it, and applies the layers in the Decoder in the correct order, while using the previously generated word as the input to the LSTM at each timestep. It also incorporates Beam Search.

visualize_att can be used to visualize the generated caption along with the weights at each timestep as seen in the examples.
