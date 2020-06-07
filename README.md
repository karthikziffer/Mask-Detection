# Face Mask Detector
Face mask classification model to detect a person wearing a mask or not. A mobilenet V2 model is trained with 97% Validation accuracy.  



### Model Description
1. The input image shape is (150, 150, 3).
2. Pre-trained Imagenet weights are employed.
3. The first 10 layers of Mobilenet V2 are frozen and the model is trained. 
4. Batch size is 16 and SGD optimizer with learning rate is 0.01
5. The Loss function is Binary cross entropy loss.


<pre style="font-family: Andale Mono, Lucida Console, Monaco, fixed, monospace; color: #000000; background-color: #eee;font-size: 12px;border: 1px dashed #999999;line-height: 14px;padding: 5px; overflow: auto; width: 100%"><code>base_model = keras.applications.MobileNetV2(
    input_shape= (150, 150, 3),
    alpha=1.0,
    include_top=False,
    weights=&quot;imagenet&quot;,
    pooling='avg')

for layer in base_model.layers[:10]:
  layer.trainable = False

x = base_model.output
out_pred = Dense(1, activation= &quot;sigmoid&quot;)(x)
model = Model(inputs = base_model.input, outputs= out_pred)
</code></pre>




### Setup

- Script: test.py
- Library Requirements: Keras, Opencv-python  


    - This script takes live video feed from openCV VideoCapture method.
    - Loads the Face Detection Model
    - Pre-process the face image  
    - Feed the image to the Mask Detection model
    - The Model classifies the outcome
    - Prints the label (Mask / No Mask ) on the image frame
    

<br>
   
### Execution

```> python test.py```

<br>

### Model Training




- Structure the Training Data Folder in the below format

```
data/
    train/
        mask/
            mask001.jpg
            mask002.jpg
            ...
        no_mask/
            no_mask001.jpg
            no_mask002.jpg
            ...
    test/
        mask/
            mask001.jpg
            mask002.jpg
            ...
        no_mask/
            no_mask001.jpg
            no_mask002.jpg
            ...
```

- Execute the train.py script


<br>

### Dataset


Real-World Masked Face Dataset

https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset


<br>

### Output Sample

![imgonline-com-ua-twotoone-L174rz-Rs-Xe59.jpg](https://i.postimg.cc/W1LwR3wS/imgonline-com-ua-twotoone-L174rz-Rs-Xe59.jpg)













