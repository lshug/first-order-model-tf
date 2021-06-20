
# first-order-model-tf
TensorFlow port of first-order motion model. TF Lite and TF.js compatible, supports the original's checkpoints and implements in-graph kp processing, but inference only (no training). 
 
Original PyTorch version can be found at [AliaksandrSiarohin/first-order-model](https://github.com/AliaksandrSiarohin/first-order-model). Copy the checkpoint tars into the checkpoint folder. If you intend to run the fashion-trained model, be sure to rename the checkpoint file for that model from fashion.pth.tar to fashion-cpk.pth.tar (the original filename for that checkpoint doesn't fit into the naming scheme for others for some reason, and that messes up the load process). Run build.py to generate saved_models and tflite files (and tf.js models if tensorflowjs_converter is installed and --tfjs flag is used). After that, you can run inference directly, with saved_models, or with tf lite, using run.py.

A colab for comparing performance and outputs between this implementation and the original is available [here](https://colab.research.google.com/drive/1CHlfG792RifIpwYQqqUNHGS_3rUe16vQ?usp=sharing).

![example](example/example.gif)

## run.py and build.py CLI
```
usage: run.py [-h] [--target {direct,savedmodel,tflite}] [--mode {animate,reconstruction}] [--datamode {file,dataset}] [--model MODEL] [--source_image SOURCE_IMAGE]
              [--driving_video DRIVING_VIDEO] [--output OUTPUT] [--dontappend] [--relative] [--adapt] [--prescale] [--frames FRAMES] [--batchsize BATCH_SIZE] [--exactbatch]
              [--device DEVICE] [--profile] [--visualizer] [--loadwithtorch]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  --target {direct,savedmodel,tflite}
                        model version to run (between running the model directly, running the model's saved_model, and running its converted tflite
  --mode {animate,reconstruction}
                        Run mode (animate, reconstruct, or train)
  --datamode {file,dataset}
                        Data input mode (CLI-given file or config-defined dataset)
  --model MODEL         model name
  --source_image SOURCE_IMAGE
                        source image path for file datamode
  --driving_video DRIVING_VIDEO
                        driving video path for file datamode
  --output OUTPUT       output file name
  --dontappend          don't append format name and .mp4 to the output filename
  --relative            relative kp mode
  --adapt               adapt movement to the proportion between the sizes of subjects in the input image and the driving video
  --prescale            Reuse the result of AntiAliasInterpolation2d performed in kp_detector in the dense motion network
  --frames FRAMES       number of frames to process
  --batchsize BATCH_SIZE
                        batch size
  --exactbatch          force static batch size, tile source image to batch size
  --device DEVICE       device to use
  --profile             enable tensorboard profiling
  --visualizer          enable visualizer, only relevant for dataset datamode
  --loadwithtorch       use torch to load checkpoints instead of trying to load tensor buffers manually (requires pytorch)
```

```
usage: build.py [-h] [--model MODEL] [-a] [--module {all,kp_detector,generator,process_kp_driving}] [--nolite] [--predictiononly] [--float16] [--tfjs]
                [--staticbatchsize STATICBATCHSIZE] [--hardcode {00,01,10,11}] [--prescale] [--loadwithtorch]

Build saved_model, tflite, and tf.js modules from checkpoints and configs.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model config and checkpoint to load
  -a                    build models for all config files
  --module {all,kp_detector,generator,process_kp_driving}
                        module to build
  --nolite              don't build tflite modules
  --predictiononly      build the generator so that it only outputs predictions
  --float16             quantize lite to float16
  --tfjs                build tf.js models, requires tensorflowjs_converter
  --staticbatchsize STATICBATCHSIZE
                        optional static batch size to use
  --hardcode {00,01,10,11}
                        optionally hardcode values for use_relative_jacobian and adapt_movement_scale at build type
  --prescale            Reuse the result of AntiAliasInterpolation2d performed in kp_detector in the dense motion network
  --loadwithtorch       use torch to load checkpoints instead of trying to load tensor buffers manually (requires pytorch)
```

## Inference details

 * First, the kps and the jacobian for the source image are detected through the kp_detector model.
 * Then, for each batch of driving video frames:
    * kps and jacobians are detected using the kp_detector model.
    * Processing of the resultant driving video frame kps and jacobians is done using process_kp_driving model (with source image and video kps/jacobians, and boolean parameters *use_relative_jacobian*, and *adapt_movement_scale* as inputs).
    * Finally, the generator model is used (with source image, source image kps/jacobian, and the video frame batch's kps/jacobians as inputs) to generate the frame predictions.
 
For more details, take a look inside animate.py or the generated tensorboard files in ./log (generated when run.py is run with --profile directly or on --target savedmodel).
 
One thing I didn't implement from the original is the find_best_frame option, which used [face-alignment](https://github.com/1adrianb/face-alignment) to find the frame in the driving video that is closest to the source image in terms of face alignment. I didn't want to include outside dependencies and this was only relevant to vox models.

## FAQ

**How do I add custom-trained models?**

Place checkpoint "{name}-cpk.pth.tar" in ./checkpoint, place "{name}-256.yml" in ./config.

**But my model uses frame shapes that are not 256x256!**

Cool. Place checkpoint "{name}-cpk.pth.tar" in ./checkpoint, place "{name}-256.yml" in ./config. 

**I'm getting weird, distorted outputs, wat do?**

This is a known issue with the reverse-engineered torch checkpoint loader in certain environments. Build/run with --loadwithtorch to use torch to load the checkpoints into numpy arrays. This requires an installation of pytorch in the environment.

**What ops are used in or necessary for the models?**

See [OPS.md](OPS.md) for the list of ops that are necessary for each of the three modules, along with notes about TF Lite delegate compatibility. The list is generated from tflite files, using tflite_ops function in utils.py. Directly built modules and SavedModels use some other ops, but those are fused/converted/erased during tflite conversion and aren't actually necessary for the model's functioning.

**What's with the weird tf.functions at the top of animate.py?**

They're there to prevent numpy-to-tensor conversion from executing eagerly. Without those, data would first be loaded into cpuland and then moved to GPU with \_Send op. That can be costly, especially in a loop.

**What would it take to add training support?**

 * Implementing the discriminator
 * Making the explicit training-disabling bools dependent on an argument in module inits. (model.trainable and trainable args in BatchNomralization inits should use the passed values; trainable should still be false for AntiAliasInterpolation2ds's kernel).
 * Implementing original's GeneratorFullModel and DiscriminatorFullModel
 * Implementing the train loop from the original's train.py, with correct handling and passing of all the items in  model config yml's train_params.
 * Adding save_weights and load_weights to the KpDetector and Generator modules for, you guessed it, saving and loading weights. Just calling the wrapped models' respective functions should work.
 
**What the fuck is going on in convex_hull_area?**

Oh boy. Long story short, it's an in-graph implementation of the gift-wrapping algorithm with all the ifs replaced with equivalent tf.where-s, followed by the shoelace formula for calculating the area, wrapped in hacks to make it work on a batch of 2d point lists all at once. Making it was fun, except when it wasn't. [Here's a pastebin of my numpy prototype](https://pastebin.com/DPPiuTip), which is way easier to read.

**What's the license/can I reuse GridSample, etc./can I use this stuff in a commercial project?**

License is MIT, so feel free to copy away to your heart's content. The original PyTorch version uses CC BY-NC 4 though, so you'll have to include an attribution thingie for it somewhere, like the one at the end of this readme. I wouldn't mind being given credit too though, so be cool and mention me somewhere.  

**Why are all of the models defined in a one giant, indecipherable file?**

I've been meaning to refactor it into multiple files, but got kinda attached to it. I wonder if people stacking shipping containers feel the same way when it's time to unstack them and send them on their way. I'm open to a pull request if anyone's willing to do the work. 

**What's with the multiple levels of function calls to generate the kp_detector/generator objects?**

Hack-around to make calling from outside not too ugly (though I guess I can't really complain about ugly code), while wrapping the keras models in modules with concrete functions for easier saved_model saving and tflite conversion. build_generator calls the constructor of the Generator tf.module, which wraps the generator keras model and the the \_\_call\_\_ of which is a tf.function with a well-defined signature which calls the model. The model itself is built by the module's constructor by running build_generator_base, which defines the keras model and loads the weights from a checkpoint file.

**Can the code be significantly optimized further?**

TensorBoard profiler and TF Lite benchmarking tool both show that almost all of the inference time is spent on conv2d ops for both generator and detector, and the time spent on process_kp_driving is already very short. If you do have any ideas, pull request away! Out of code, all the usual post-training tf lite optimizations can be added to reduce .tflite sizes and to speed up inference. Refer to [TF Lite optimization guide](https://www.tensorflow.org/lite/performance/model_optimization) and to [tensorflow-model-optimization documentation's weight clustering guide](https://www.tensorflow.org/model_optimization/guide/clustering/clustering_example).

**Could I make a single keras model that receives source image and driving video and outputs the predicted video?**

Well yes, but actually no. It would certainly be possible to make a model subclass that does just that, but doing so with a vanilla model will all but make it necessary to match the batch sizes of the source image and the driving video. Since the architecture uses the driving video frames as batches (i.e. each batch contains driving video frames, as opposed to each batch containing several complete driving videos), this would make it necessary to tile the source image to the batch dimension of the driving video and process the source image and source image kps/jacobian the number of times equal to the number of frames in the video. This is very inefficient. As for creating a single tf.Module or a model subclass that handles the full inference pipeline that's handled in animate.py, the main reasons I'm not doing that is that without the control flow that tf lite conversion cannot handle, the only possible way to run the model as a single callable would be to pass the entirety of kp_driving and kp_driving_jacobian to the generator as a single video-length batch, which would have debilitating impact on memory and performance. To get a feel of how this would run, run run.py with batch size equal to the number of frames in the driving video. 

**Could I use the keras models for generator and kp_detector directly without using the tf.Module wrappers?**

Yes. You can do that by calling build\_generator\_base and build_kp_detector_base directly, but saving and loading will be a hassle because of all the custom objects.
 
**Why would anyone willingly do this?**

¯\\\_(ツ)\_/¯
<details>
  <summary>...</summary>
  
>So, the full story is, I was in an emergency situation financially in Spring 2020 while finishing my last university semester, so I registered on Upwork to work for unreasonably low pay. I agreed to do this project for $50 under a one week deadline. I couldn't make the deadline, and got my first big panic attack as a reward. I couldn't get myself to contact the employer, and I desperately tried to finish the project, adding and optimizing one part after another. By the time I got it to a state in which I could hand it in as a deliverable (pretty close to the initial commit of this repo), too much time had passed and I couldn't make myself open the messages and write the guys about the situations. At the end just logging in or looking and this code was giving me anxiety, and I made my sister close my Upwork account and dropped this project, along with one other. Then a year passed, and a friend asked me to do a funny motion transfer video to make a still image of David Beckham talk for my sister's birthday, and I did have the code laying around and didn't want to pull the original repo again, so I did it with my version. Then I looked through the code and realized that this was in no way a $50, one-week project, and that I could be proud of myself for doing it regardless of the outcome. So I made this repo and open-sourced the code. There's a lesson to be learned here, but I feel like I'm the only one who needed to learn it. And if a potential employer is reading this, uhh, just ignore this whole paragraph, everyone has their highs and lows and all.
</details>

## Bragging section

Boy, was making this thing work with the original's checkpoints with tf lite and with >1 frame batches a journey. Some stuff I had to do to achieve that:

 * Translate the internals of PyTorch's bilinear interpolation op into tf code
 * Same with nearest-interpolation
 * Same with F.grid_sample
 * Implement a tf lite-compatible in-graph calculation of the area of a 2D convex hull given the number of points (for processing kps in-graph; original uses scipy.spatial, which itself uses qhull, and I wanted everything to be handled in-graph so that the three tf lite models would be able to handle the full inference pipeline from the source image and the driving video all the way to the inferred video)
 * Translate numpy-like indexings into equivalent tf.gather_nd calls
 * Translate all the weird little constructs of the original into keras layers (all the stuff used in the dense motion network module in particular made me cry a few times)
 * Get rid of all the autoamtic shape broadcasting that'd make life a little easier because apparently they make tf lite crash
 * Do some seriously complicated math to derive equivalent sequences of tensor reshapes and transpositions
 * Take all those tf ops that are really pretty and elegant and replace them with esoteric magic so that tf lite doesn't complain
 * Export intermediate outputs during inference at two dozen places in both tf and torch code to manually compare differences until they evaporated
 * Reverse-engineer pytorch's checkpoint loader (this wasn't strictly necessary, and earlier I just used torch.load to load the checkpoints, but having torch as a requirement in a tf port of a torch project seemed in bad taste)
 * Some other stuff I barely remember

In the end, it actually turned out a little faster than the original. Kudos to me.

### Attribution
[first-order-model](https://github.com/AliaksandrSiarohin/first-order-model) by [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin), used under [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) / Ported to TensorFlow
